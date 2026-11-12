"""AdamBridge inside an unmodified Hugging Face Trainer (gradient accumulation included)."""

import tempfile

import pytest
import torch

import adambridge
from conftest import max_param_diff

transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.cuda


def _train(make_opt, accumulation: int):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).cuda()
    g = torch.Generator().manual_seed(0)
    data = [{"input_ids": ids, "labels": ids} for ids in torch.randint(0, 128, (32, 16), generator=g)]
    with tempfile.TemporaryDirectory() as tmp:
        args = transformers.TrainingArguments(
            output_dir=tmp, per_device_train_batch_size=4, gradient_accumulation_steps=accumulation, max_steps=4,
            learning_rate=1e-3, lr_scheduler_type="constant", weight_decay=0.0, max_grad_norm=0.0, report_to=[],
            logging_strategy="no", save_strategy="no", seed=0, data_seed=0, dataloader_pin_memory=False,
            disable_tqdm=True)
        opt = make_opt(model, accumulation)
        transformers.Trainer(model=model, args=args, train_dataset=data, optimizers=(opt, None)).train()
    if isinstance(opt, adambridge.AdamBridge):
        opt.close()
    return model


@pytest.mark.parametrize("accumulation", [1, 2])
@pytest.mark.parametrize("delay", [False, True])
def test_trainer_matches_torch_adamw(accumulation, delay):
    ref = _train(lambda m, a: torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0), accumulation)
    if delay:
        # Delayed updates are one step stale by design: compare loosely (they still train the same model).
        model = _train(lambda m, a: adambridge.AdamBridge(m.parameters(), lr=1e-3, weight_decay=0.0, delay_updates=True,
                                                          accumulation_steps=a).attach_hooks(m), accumulation)
        assert max_param_diff(ref, model) < 5e-3
        return
    model = _train(lambda m, a: adambridge.AdamBridge(m.parameters(), lr=1e-3, weight_decay=0.0,
                                                      accumulation_steps=a).attach_hooks(m), accumulation)
    assert max_param_diff(ref, model) < 1e-5
