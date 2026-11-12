"""gpu_state_budget: the optimizer state of the last parameters in backward kept on the GPU."""

import copy

import pytest
import torch

import adambridge
from conftest import max_param_diff
from test_cuda import TinyLM, lm_loss

pytestmark = pytest.mark.cuda


def _train(model, opt, steps, start=0, hooks=True):
    for k in range(start, start + steps):
        lm_loss(model, k).backward()
        if hooks:
            opt.step_sync()
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
    opt.flush()
    torch.cuda.synchronize()


def _reference(model, steps):
    ref = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for k in range(steps):
        ref.zero_grad()
        lm_loss(model, k).backward()
        ref.step()
    return model


@pytest.mark.parametrize("budget", [0, "half", "all", "auto"])
@pytest.mark.parametrize("hooks", [True, False])
def test_matches_torch_adamw(budget, hooks):
    torch.manual_seed(0)
    base = TinyLM().cuda()
    total = sum(p.numel() for p in base.parameters()) * 18
    value = {"half": total // 2, "all": total * 2}.get(budget, budget)
    model = copy.deepcopy(base)
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3, small_param_threshold=2000, gpu_state_budget=value)
    if hooks:
        opt.attach_hooks(model)
    _train(model, opt, 8, hooks=hooks)
    ref = _reference(copy.deepcopy(base), 8)
    on_gpu = [s for s in opt._ordered if s.gpu is not None]
    large = [s for s in opt._ordered if s.small is None]
    if budget == 0:
        assert not on_gpu
    elif budget == "half":
        assert 0 < len(on_gpu) < len(large)
    elif budget in ("all", "auto"):
        assert len(on_gpu) == len(large)  # TinyLM fits easily
    if hooks and budget == "half":
        # The parameters kept on the GPU are the last to get their gradients: the embedding (tied with the head,
        # its gradient completes last) and the first block's.
        assert opt._slots[model.emb.weight].gpu is not None
    report = opt.memory_report()
    assert report["gpu_state"] == sum(s.numel * 12 for s in on_gpu)
    assert max_param_diff(ref, model) < 1e-5
    opt.close()


def test_checkpoint_round_trip_across_budgets():
    """A checkpoint saved with state on the GPU loads into an optimizer without, and the reverse; training then
    continues exactly as without the round trip."""
    torch.manual_seed(0)
    base = TinyLM().cuda()
    a = copy.deepcopy(base)
    opt_a = adambridge.AdamBridge(a.parameters(), lr=3e-3, small_param_threshold=2000,
                                  gpu_state_budget=1 << 40).attach_hooks(a)
    _train(a, opt_a, 4)
    sd = opt_a.state_dict()
    b = copy.deepcopy(a)
    opt_b = adambridge.AdamBridge(b.parameters(), lr=3e-3, small_param_threshold=2000).attach_hooks(b)
    opt_b.load_state_dict(sd)
    _train(a, opt_a, 4, start=4)
    _train(b, opt_b, 4, start=4)
    assert max_param_diff(a, b) < 1e-6
    # and back: host state into an optimizer that keeps state on the GPU
    c = copy.deepcopy(b)
    opt_c = adambridge.AdamBridge(c.parameters(), lr=3e-3, small_param_threshold=2000,
                                  gpu_state_budget=1 << 40).attach_hooks(c)
    _train(c, opt_c, 1, start=8)          # places the state on the GPU after its first step
    opt_c.load_state_dict(opt_b.state_dict())
    with torch.no_grad():
        for p, q in zip(c.parameters(), b.parameters()):
            p.copy_(q)
    _train(b, opt_b, 3, start=8)
    _train(c, opt_c, 3, start=8)
    assert max_param_diff(b, c) < 1e-6
    for opt in (opt_a, opt_b, opt_c):
        opt.close()


def test_with_delayed_updates_and_accumulation():
    """State on the GPU is updated exactly at step_sync, alongside delayed host updates and accumulation."""
    torch.manual_seed(0)
    model = TinyLM().cuda()
    total = sum(p.numel() for p in model.parameters()) * 18
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3, small_param_threshold=2000, delay_updates=True,
                                accumulation_steps=2, gpu_state_budget=total // 3).attach_hooks(model)
    for k in range(6):
        for micro in range(2):
            lm_loss(model, 10 * k + micro).backward()
        opt.step_sync()
    opt.flush()
    assert any(s.gpu is not None for s in opt._ordered) and any(s.gpu is None and s.small is None
                                                                for s in opt._ordered)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    opt.close()


def test_rejects_lion_and_bad_values():
    p = [torch.nn.Parameter(torch.zeros(4, device="cuda"))]
    with pytest.raises(ValueError):
        adambridge.LionBridge(p, gpu_state_budget="auto")
    with pytest.raises(ValueError):
        adambridge.AdamBridge(p, gpu_state_budget="lots")
