import copy
import io

import pytest
import torch

import adambridge
from conftest import BACKENDS, max_param_diff, tiny_mlp


def train(model, opt, steps, seed=1, batch=16, hooks=False, device="cpu"):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(batch, 48, device=device, dtype=next(model.parameters()).dtype)
        opt.zero_grad()
        loss = model(x).float().pow(2).mean()
        loss.backward()
        if hooks:
            opt.step_sync()
        else:
            opt.step()
        losses.append(loss.item())
    return losses


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_torch_adamw_with_groups_and_scheduler(backend):
    ref_model = tiny_mlp()
    model = copy.deepcopy(ref_model)

    def groups(m):
        return [
            {"params": [m[0].weight, m[3].weight], "weight_decay": 0.1},
            {"params": [m[0].bias, m[2].weight, m[2].bias, m[3].bias], "weight_decay": 0.0, "lr": 3e-3},
        ]

    ref = torch.optim.AdamW(groups(ref_model), lr=1e-2, betas=(0.8, 0.99))
    opt = adambridge.AdamBridge(groups(model), lr=1e-2, betas=(0.8, 0.99), backend=backend)
    sched_ref = torch.optim.lr_scheduler.StepLR(ref, step_size=3, gamma=0.5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    torch.manual_seed(1)
    for _ in range(10):
        x = torch.randn(16, 48)
        for m, o, s in ((ref_model, ref, sched_ref), (model, opt, sched)):
            o.zero_grad()
            m(x).pow(2).mean().backward()
            o.step()
            s.step()
    assert max_param_diff(ref_model, model) < 1e-6
    opt.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_cpu_bf16_params_use_fp32_master(backend):
    ref_model = tiny_mlp()
    model = copy.deepcopy(ref_model).to(torch.bfloat16)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-3)
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-3, backend=backend)
    torch.manual_seed(1)
    for _ in range(5):
        x = torch.randn(16, 48)
        ref.zero_grad()
        ref_model(x).pow(2).mean().backward()
        # Feed both optimizers the same gradients, so only master precision differs.
        for p, q in zip(ref_model.parameters(), model.parameters()):
            q.grad = p.grad.to(torch.bfloat16)
        ref.step()
        opt.step()
    for p, q, master in zip(ref_model.parameters(), model.parameters(), opt.master_params()):
        assert q.dtype == torch.bfloat16
        torch.testing.assert_close(master.view_as(p), p.detach(), rtol=1e-2, atol=1e-4)
        assert torch.equal(q, master.view_as(q).to(torch.bfloat16))
    opt.close()


def test_params_without_grad_are_skipped():
    model = tiny_mlp()
    frozen = model[0].weight.detach().clone()
    model[0].weight.requires_grad_(False)
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-2)
    train(model, opt, 2)
    assert torch.equal(model[0].weight, frozen)
    assert opt.master_params()[0] is None
    opt.close()


def test_state_dict_resume_is_exact():
    torch.manual_seed(0)
    base = tiny_mlp()
    a = copy.deepcopy(base)
    opt_a = adambridge.AdamBridge(a.parameters(), lr=1e-2, precision="bf16", seed=7)
    train(a, opt_a, 6, seed=3)

    b = copy.deepcopy(base)
    opt_b = adambridge.AdamBridge(b.parameters(), lr=1e-2, precision="bf16", seed=7)
    train(b, opt_b, 3, seed=3)
    buf = io.BytesIO()
    torch.save({"model": b.state_dict(), "opt": opt_b.state_dict()}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)

    c = tiny_mlp(seed=123)  # different init, overwritten by the checkpoint
    opt_c = adambridge.AdamBridge(c.parameters(), lr=1e-2, precision="bf16", seed=7)
    c.load_state_dict(ckpt["model"])
    opt_c.load_state_dict(ckpt["opt"])
    # continue with the same data stream as `a` after step 3
    torch.manual_seed(3)
    for _ in range(3):
        torch.randn(16, 48)
    for _ in range(3):
        x = torch.randn(16, 48)
        opt_c.zero_grad()
        c(x).pow(2).mean().backward()
        opt_c.step()
    assert max_param_diff(a, c) == 0
    for o in (opt_a, opt_b, opt_c):
        o.close()


@pytest.mark.parametrize("prec", ["fp32", "fp8"])
def test_load_torch_adamw_checkpoint(prec):
    ref_model = tiny_mlp()
    ref = torch.optim.AdamW(ref_model.parameters(), lr=1e-2)
    train(ref_model, ref, 3)
    model = copy.deepcopy(ref_model)
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-2, precision=prec)
    opt.load_state_dict(ref.state_dict())
    for slot, p in zip(opt._ordered, ref_model.parameters()):
        assert slot.step == 3 and slot.m.precision == prec
        m_ref = ref.state[p]["exp_avg"].reshape(-1)
        tol = 1e-7 if prec == "fp32" else 0.07 * m_ref.abs().max().item()  # E4M3: 3 mantissa bits
        assert (slot.m.to_fp32() - m_ref).abs().max().item() <= tol
    train(ref_model, ref, 3, seed=5)
    train(model, opt, 3, seed=5)
    # Weights move ~3e-2 over these steps; FP8 moments perturb the step size.
    assert max_param_diff(ref_model, model) < (1e-6 if prec == "fp32" else 1e-2)
    opt.close()


def test_weights_loaded_after_construction_are_respected():
    """Master weights are created lazily, so loading pretrained weights after
    building the optimizer is not silently reverted on the first step."""
    model = tiny_mlp(seed=0)
    opt = adambridge.AdamBridge(model.parameters(), lr=0.0, weight_decay=0.0)
    pretrained = tiny_mlp(seed=42)
    model.load_state_dict(pretrained.state_dict())
    train(model, opt, 1)
    assert max_param_diff(model, pretrained) == 0
    opt.close()


@pytest.mark.parametrize("hooks", [False, True])
def test_set_precision_mid_training(hooks):
    model = tiny_mlp()
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-2)
    if hooks:
        opt.attach_hooks(model)
    train(model, opt, 2, hooks=hooks)
    m_before = [s.m.to_fp32() for s in opt._ordered]
    opt.set_precision("bf16_moments")
    for s, m in zip(opt._ordered, m_before):
        assert s.m.precision == "bf16"
        torch.testing.assert_close(s.m.to_fp32(), m, rtol=2**-8, atol=0)
    train(model, opt, 2, hooks=hooks)
    opt.close()


@pytest.mark.parametrize("prec", ["bf16", "fp8"])
def test_low_precision_moments_converge_like_fp32(prec):
    torch.manual_seed(0)
    w_true = torch.randn(64, 16)
    x_all = torch.randn(2048, 64)
    y_all = x_all @ w_true + 0.01 * torch.randn(2048, 16)

    def run(precision):
        torch.manual_seed(1)
        lin = torch.nn.Linear(64, 16)
        opt = adambridge.AdamBridge(lin.parameters(), lr=1e-2, weight_decay=0.0, precision=precision)
        for i in range(1200):
            idx = torch.randint(0, 2048, (64,))
            opt.zero_grad()
            loss = (lin(x_all[idx]) - y_all[idx]).pow(2).mean()
            loss.backward()
            opt.step()
        opt.close()
        return (lin(x_all) - y_all).pow(2).mean().item()

    base, low = run("fp32"), run(prec)
    assert base < 2e-4  # noise floor is 1e-4
    assert low < 1.2 * base


def test_double_backward_without_no_sync_raises():
    model = tiny_mlp()
    opt = adambridge.AdamBridge(model.parameters(), release_grads=False).attach_hooks(model)
    x = torch.randn(4, 48)
    model(x).sum().backward()
    with pytest.raises(RuntimeError, match="no_sync"):
        model(x).sum().backward()
    opt.step_sync()
    opt.close()


def test_hooks_on_cpu_match_step():
    a, b = tiny_mlp(), tiny_mlp()
    opt_a = adambridge.AdamBridge(a.parameters(), lr=1e-2)
    opt_b = adambridge.AdamBridge(b.parameters(), lr=1e-2).attach_hooks(b)
    train(a, opt_a, 4)
    train(b, opt_b, 4, hooks=True)
    assert max_param_diff(a, b) == 0
    assert all(p.grad is None for p in b.parameters())  # released after offload
    opt_a.close()
    opt_b.close()


def test_worker_errors_surface_in_step():
    model = tiny_mlp()
    opt = adambridge.AdamBridge(model.parameters())
    model(torch.randn(2, 48)).sum().backward()
    opt.param_groups[0]["betas"] = None  # makes the worker's update raise
    with pytest.raises(RuntimeError, match="host update failed"):
        opt.step()
    opt.close()


def test_memory_report_and_backend():
    model = tiny_mlp()
    n = sum(p.numel() for p in model.parameters())
    for prec, per in (("fp32", 8), ("bf16", 4)):
        opt = adambridge.AdamBridge(model.parameters(), precision=prec)
        train(model, opt, 1)
        rep = opt.memory_report()
        assert rep["master"] == 0  # CPU FP32 params are updated in place
        assert rep["moments"] == per * n
        assert opt.backend.startswith(("native/", "torch"))
        opt.close()


def test_alias():
    assert adambridge.OptiComputeAdamW is adambridge.AdamBridge
