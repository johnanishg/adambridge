"""causal_lm_loss: the same value and gradients as the Hugging Face loss, without the full logits tensor."""

import pytest
import torch

import adambridge
from conftest import max_param_diff

transformers = pytest.importorskip("transformers")

VOCAB = 256


def _model(seed=0, vocab=VOCAB, hidden=64, device="cpu", dtype=torch.float32, tied=True):
    torch.manual_seed(seed)
    cfg = transformers.Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=1024, tie_word_embeddings=tied)
    return transformers.Qwen2ForCausalLM(cfg).to(device=device, dtype=dtype)


def _ids(shape, seed=1, device="cpu"):
    return torch.randint(0, VOCAB, shape, generator=torch.Generator().manual_seed(seed)).to(device)


def _reference(model, ids, labels):
    loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    grads = [q.grad.clone() for q in model.parameters()]
    model.zero_grad(set_to_none=True)
    return loss.detach(), grads


def _chunked(model, ids, labels, chunk):
    hidden = model.model(input_ids=ids).last_hidden_state
    loss = adambridge.causal_lm_loss(hidden, model.lm_head, labels, chunk_tokens=chunk)
    loss.backward()
    grads = [q.grad.clone() for q in model.parameters()]
    model.zero_grad(set_to_none=True)
    return loss.detach(), grads


@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("chunk", [1, 7, 16, 1000])
def test_matches_transformers_loss_and_gradients(chunk, tied):
    model = _model(tied=tied)
    ids = _ids((3, 16))
    ref, ref_grads = _reference(model, ids, ids)
    ours, grads = _chunked(model, ids, ids, chunk)
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_ignored_labels_are_left_out_like_transformers():
    model = _model()
    ids = _ids((2, 12))
    labels = ids.clone()
    labels[0, :5] = -100   # a prompt that is not scored
    labels[1, 8:] = -100   # padding
    ref, ref_grads = _reference(model, ids, labels)
    ours, grads = _chunked(model, ids, labels, 5)
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_all_labels_ignored_gives_zero_not_nan():
    model = _model()
    ids = _ids((1, 4))
    hidden = model.model(input_ids=ids).last_hidden_state
    loss = adambridge.causal_lm_loss(hidden, model.lm_head, torch.full_like(ids, -100), chunk_tokens=2)
    assert loss.item() == 0.0


def test_no_grad_evaluation_matches():
    model = _model().eval()
    ids = _ids((2, 16))
    with torch.no_grad():
        ref = model(input_ids=ids, labels=ids).loss
        hidden = model.model(input_ids=ids).last_hidden_state
        ours = adambridge.causal_lm_loss(hidden, model.lm_head, ids, chunk_tokens=6)
    assert not ours.requires_grad
    torch.testing.assert_close(ours, ref, rtol=1e-6, atol=1e-6)


def test_bad_arguments_are_rejected():
    model = _model()
    ids = _ids((2, 8))
    hidden = model.model(input_ids=ids).last_hidden_state
    with pytest.raises(ValueError, match="labels"):
        adambridge.causal_lm_loss(hidden, model.lm_head, ids[:, :4])
    with pytest.raises(ValueError, match="hidden_states"):
        adambridge.causal_lm_loss(hidden[0], model.lm_head, ids)
    for chunk in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="chunk_tokens"):
            adambridge.causal_lm_loss(hidden, model.lm_head, ids, chunk_tokens=chunk)


def test_adambridge_hooks_see_one_gradient_per_parameter():
    """The tied lm_head/embedding weight is used in every slice and again when the slices are recomputed in the
    backward pass: still one accumulated gradient per parameter, so the hooks fire once (no 'second gradient')."""
    a, b = _model(tied=True), _model(tied=True)
    opt_a = torch.optim.AdamW(a.parameters(), lr=1e-3, weight_decay=0.0)
    opt_b = adambridge.AdamBridge(b.parameters(), lr=1e-3, weight_decay=0.0, small_param_threshold=0).attach_hooks(b)
    for step in range(3):
        ids = _ids((2, 16), seed=10 + step)
        for model, opt in ((a, opt_a), (b, opt_b)):
            hidden = model.model(input_ids=ids).last_hidden_state
            adambridge.causal_lm_loss(hidden, model.lm_head, ids, chunk_tokens=8).backward()
            if isinstance(opt, adambridge.AdamBridge):
                opt.step_sync()
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)
    opt_b.close()
    assert max_param_diff(a, b) < 1e-5


@pytest.mark.cuda
def test_peak_memory_is_a_fraction_of_the_full_logits():
    model = _model(vocab=32768, device="cuda", dtype=torch.bfloat16)
    ids = _ids((4, 512), device="cuda")  # full logits: 2048 x 32768 x 4 B = 256 MiB in FP32, held twice

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        fn().backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - before) / 2**20

    full = peak(lambda: model(input_ids=ids, labels=ids).loss)
    chunked = peak(lambda: adambridge.causal_lm_loss(model.model(input_ids=ids).last_hidden_state, model.lm_head, ids,
                                                     chunk_tokens=256))
    assert chunked < full / 2, (full, chunked)
