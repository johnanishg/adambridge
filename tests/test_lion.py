"""LionBridge on the CPU: against a plain PyTorch Lion, every moment precision,
both backends, checkpoints and memory accounting."""

import copy
import io

import pytest
import torch

import adambridge
from conftest import BACKENDS, max_param_diff, tiny_mlp


class RefLion(torch.optim.Optimizer):
    """Lion as published (decoupled weight decay, sign of the interpolated moment)."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                m = self.state[p].setdefault("m", torch.zeros_like(p))
                c = m * b1 + p.grad * (1 - b1)
                p.mul_(1 - group["lr"] * group["weight_decay"]).add_(c.sign(), alpha=-group["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


def run(model, opt, steps, seed=1):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(16, 48)
        opt.zero_grad()
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_reference_lion_with_groups(backend):
    ref_model = tiny_mlp()
    model = copy.deepcopy(ref_model)

    def groups(m):
        return [{"params": [m[0].weight, m[3].weight], "weight_decay": 0.5},
                {"params": [m[0].bias, m[2].weight, m[2].bias, m[3].bias], "lr": 3e-4}]

    ref = RefLion(groups(ref_model), lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    opt = adambridge.LionBridge(groups(model), lr=1e-3, betas=(0.9, 0.99), backend=backend)
    run(ref_model, ref, 10)
    run(model, opt, 10)
    assert max_param_diff(ref_model, model) < 1e-6
    opt.close()


@pytest.mark.parametrize("prec", ["bf16", "fp8"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_low_precision_moment_trains_like_fp32(prec, backend):
    results = {}
    for p in ("fp32", prec):
        model = tiny_mlp()
        opt = adambridge.LionBridge(model.parameters(), lr=3e-4, precision=p, backend=backend)
        results[p] = run(model, opt, 60)
        opt.close()
    assert results[prec][-1] < 0.5 * results[prec][0]
    assert abs(results[prec][-1] - results["fp32"][-1]) < 0.1 * results["fp32"][0]


def test_native_and_torch_backends_agree():
    if "native" not in BACKENDS:
        pytest.skip("native kernels unavailable")
    models = {}
    for backend in ("torch", "native"):
        model = models[backend] = tiny_mlp()
        opt = adambridge.LionBridge(model.parameters(), lr=1e-3, weight_decay=0.1, backend=backend)
        run(model, opt, 8)
        opt.close()
    assert max_param_diff(models["torch"], models["native"]) < 1e-6


def test_one_moment_in_memory_and_checkpoint():
    model = tiny_mlp()
    opt = adambridge.LionBridge(model.parameters(), lr=1e-3)
    run(model, opt, 2)
    n = sum(p.numel() for p in model.parameters())
    assert opt.memory_report()["moments"] == 4 * n
    assert opt.expected_bytes_per_param() == 8.0
    assert adambridge.bytes_per_param("bf16", moments=1) == 6.0
    sd = opt.state_dict()
    assert sd["adambridge"]["algorithm"] == "lion"
    assert all("exp_avg" in st and "exp_avg_sq" not in st for st in sd["state"].values())
    opt.close()


def test_state_dict_resume_is_exact():
    model = tiny_mlp()
    opt = adambridge.LionBridge(model.parameters(), lr=1e-3, precision="bf16")
    run(model, opt, 3)
    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, buf)
    run(model, opt, 4, seed=7)
    expected = [p.detach().clone() for p in model.parameters()]

    buf.seek(0)
    ckpt = torch.load(buf)
    model2 = tiny_mlp(seed=5)
    model2.load_state_dict(ckpt["model"])
    opt2 = adambridge.LionBridge(model2.parameters(), lr=1e-3, precision="bf16")
    opt2.load_state_dict(ckpt["opt"])
    run(model2, opt2, 4, seed=7)
    assert all(torch.equal(a, b) for a, b in zip(expected, model2.parameters()))

    adam = adambridge.AdamBridge(tiny_mlp().parameters())
    with pytest.raises(ValueError, match="exp_avg_sq"):
        adam.load_state_dict(ckpt["opt"])
    for o in (opt, opt2, adam):
        o.close()


def test_no_eps_argument():
    with pytest.raises(TypeError):
        adambridge.LionBridge(tiny_mlp().parameters(), eps=1e-8)


@pytest.mark.parametrize("window", [3, 2])
def test_accumulation_steps_match_accumulated_adamw(window):
    """Hooks fire on every backward; ``accumulation_steps`` holds the update
    until the last micro-batch (3 of 3), or step_sync picks up a window cut
    short (2 of 3)."""
    ref_model = tiny_mlp()
    model = copy.deepcopy(ref_model)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-2, accumulation_steps=3).attach_hooks(model)
    torch.manual_seed(1)
    for _ in range(4):
        xs = [torch.randn(8, 48) for _ in range(window)]
        ref.zero_grad()
        for x in xs:
            (ref_model(x).pow(2).mean() / window).backward()
        ref.step()
        for x in xs:
            (model(x).pow(2).mean() / window).backward()
        opt.step_sync()
        opt.zero_grad()
    assert max_param_diff(ref_model, model) < 1e-6
    opt.close()
