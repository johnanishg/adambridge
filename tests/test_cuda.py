"""The asynchronous offload pipeline on a real GPU."""

import copy

import pytest
import torch

import adamx
from conftest import BACKENDS, max_param_diff, tiny_mlp

pytestmark = pytest.mark.cuda


class TinyLM(torch.nn.Module):
    """Embedding + blocks + tied output head: several layers, weight tying."""

    def __init__(self, vocab=97, dim=64, layers=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, 4 * dim), torch.nn.GELU(), torch.nn.Linear(4 * dim, dim))
            for _ in range(layers)
        )
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, tokens):
        h = self.emb(tokens)
        for blk in self.blocks:
            h = h + blk(h)
        return self.norm(h) @ self.emb.weight.t()


def lm_loss(model, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens = torch.randint(0, 97, (8, 33), device="cuda", generator=g)
    logits = model(tokens[:, :-1])
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("overlap", [True, False])
def test_async_hooks_match_torch_adamw(backend, overlap):
    torch.manual_seed(0)
    ref_model = TinyLM().cuda()
    model = copy.deepcopy(ref_model)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=3e-3)
    opt = adamx.AdamX(model.parameters(), lr=3e-3, backend=backend, overlap_writeback=overlap).attach_hooks(model)
    for step in range(8):
        ref.zero_grad()
        lm_loss(ref_model, step).backward()
        ref.step()
        opt.zero_grad()
        lm_loss(model, step).backward()
        assert all(p.grad is None for p in model.parameters())
        stats = opt.step_sync()
        assert stats.num_updates == len(list(model.parameters()))
    assert max_param_diff(ref_model, model) < 1e-5
    opt.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_sync_step_on_cuda_with_clipping(backend):
    torch.manual_seed(0)
    ref_model = TinyLM().cuda()
    model = copy.deepcopy(ref_model)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=3e-3)
    opt = adamx.AdamX(model.parameters(), lr=3e-3, backend=backend)
    for step in range(5):
        for m, o in ((ref_model, ref), (model, opt)):
            o.zero_grad()
            lm_loss(m, step).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.1)
            o.step()
    assert max_param_diff(ref_model, model) < 1e-5
    opt.close()


def test_bf16_params_on_gpu():
    """GPU holds BF16 weights; the host keeps FP32 masters and moments."""
    torch.manual_seed(0)
    model = TinyLM().cuda().to(torch.bfloat16)
    opt = adamx.AdamX(model.parameters(), lr=3e-3, precision="bf16").attach_hooks(model)
    losses = []
    for step in range(30):
        loss = lm_loss(model, step % 4)
        loss.backward()
        opt.step_sync()
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 0.5
    for p, master in zip(model.parameters(), opt.master_params()):
        assert p.dtype == torch.bfloat16 and master.dtype == torch.float32
        assert torch.equal(p.cpu().reshape(-1), master.to(torch.bfloat16))
    rep = opt.memory_report()
    n = sum(p.numel() for p in model.parameters())
    assert rep["master"] == 4 * n and rep["moments"] == 4 * n
    opt.close()


def test_gradient_accumulation_with_no_sync():
    torch.manual_seed(0)
    ref_model = TinyLM().cuda()
    model = copy.deepcopy(ref_model)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=3e-3)
    opt = adamx.AdamX(model.parameters(), lr=3e-3).attach_hooks(model)
    for step in range(3):
        ref.zero_grad()
        for micro in range(3):
            (lm_loss(ref_model, 10 * step + micro) / 3).backward()
        ref.step()
        for micro in range(3):
            if micro < 2:
                with opt.no_sync():
                    (lm_loss(model, 10 * step + micro) / 3).backward()
            else:
                (lm_loss(model, 10 * step + micro) / 3).backward()
        opt.step_sync()
    assert max_param_diff(ref_model, model) < 1e-5
    opt.close()


def test_weights_visible_to_next_forward_without_host_sync():
    """step_sync orders the write-back on the current stream; no device-wide sync needed."""
    torch.manual_seed(0)
    model = tiny_mlp(device="cuda")
    opt = adamx.AdamX(model.parameters(), lr=0.5).attach_hooks(model)
    x = torch.randn(4, 48, device="cuda")
    model(x).sum().backward()
    opt.step_sync()
    after = [p.clone() for p in model.parameters()]  # enqueued on the current stream
    torch.cuda.synchronize()
    for p, q in zip(model.parameters(), after):
        assert torch.equal(p, q)
    host = [m.view_as(p) for p, m in zip(model.parameters(), opt.master_params())]
    for q, m in zip(after, host):
        assert torch.equal(q.cpu(), m)
    opt.close()


def test_autotune_runs_and_records_decisions():
    torch.manual_seed(0)
    model = TinyLM().cuda()
    cfg = adamx.AutoTuneConfig(warmup_steps=1, window=2, threshold=0.0, min_precision="bf16")
    opt = adamx.AdamX(model.parameters(), lr=1e-3, autotune=cfg).attach_hooks(model)
    for step in range(12):
        lm_loss(model, step).backward()
        stats = opt.step_sync()
    assert stats.gpu_backward is not None and stats.gpu_backward > 0
    assert opt._tuner.threads_tuned
    # threshold=0 forces "host-bound", so the tuner must at least try bf16
    assert any("bf16" in d for d in opt.autotune_log)
    assert opt.precision in ("fp32", "bf16")
    opt.close()


def test_mixed_cpu_and_cuda_params():
    gpu = torch.nn.Linear(8, 8).cuda()
    cpu = torch.nn.Linear(8, 8)
    opt = adamx.AdamX([*gpu.parameters(), *cpu.parameters()], lr=1e-2)
    before = [p.detach().clone() for p in (*gpu.parameters(), *cpu.parameters())]
    gpu(torch.randn(2, 8, device="cuda")).sum().backward()
    cpu(torch.randn(2, 8)).sum().backward()
    opt.step()
    after = [p.detach() for p in (*gpu.parameters(), *cpu.parameters())]
    assert all(not torch.equal(a.cpu(), b.cpu()) for a, b in zip(before, after))
    opt.close()
