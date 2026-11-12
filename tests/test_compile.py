"""adambridge's hooks under torch.compile: whole-model compile in every mode
(CUDA graphs included), per-block compile, compiled autograd, and delayed
updates. Needs CUDA and Triton."""

import copy

import pytest
import torch

import adambridge
from conftest import max_param_diff
from test_cuda import TinyLM, _stale_reference, lm_loss

pytestmark = pytest.mark.cuda
pytest.importorskip("triton", reason="torch.compile needs Triton (not installed)")


def per_block(model):
    for blk in model.blocks:
        blk.compile()
    return model


COMPILERS = {
    "default": lambda m: torch.compile(m),
    "reduce-overhead": lambda m: torch.compile(m, mode="reduce-overhead"),  # CUDA graphs
    "max-autotune": lambda m: torch.compile(m, mode="max-autotune"),
    "per-block": per_block,
}


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _train(fwd, model, opt, steps=6, compiled_autograd=False):
    for step in range(steps):
        loss = lm_loss(fwd, step)
        if compiled_autograd:
            with torch._dynamo.compiled_autograd._enable(torch.compile):
                loss.backward()
        else:
            loss.backward()
        opt.step_sync()


def _reference(model, steps=6):
    ref_model = copy.deepcopy(model)
    ref = torch.optim.AdamW(ref_model.parameters(), lr=3e-3)
    for step in range(steps):
        ref.zero_grad()
        lm_loss(ref_model, step).backward()
        ref.step()
    return ref_model


@pytest.mark.parametrize("how", list(COMPILERS))
def test_compiled_model_matches_torch_adamw(how):
    torch.manual_seed(0)
    model = TinyLM().cuda()
    ref_model = _reference(model)
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3, small_param_threshold=0).attach_hooks(model)
    _train(COMPILERS[how](model), model, opt)
    assert max_param_diff(ref_model, model) < 1e-5
    opt.close()


def test_compiled_autograd_matches_torch_adamw():
    """Compiled autograd traces the backward pass, hooks included: the hook
    must stay opaque to Dynamo (it is called, not traced)."""
    torch.manual_seed(0)
    model = TinyLM().cuda()
    ref_model = _reference(model)
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3).attach_hooks(model)
    _train(torch.compile(model), model, opt, compiled_autograd=True)
    assert max_param_diff(ref_model, model) < 1e-5
    opt.close()


@pytest.mark.parametrize("how", ["per-block", "reduce-overhead"])
def test_delayed_updates_under_compile(how):
    torch.manual_seed(0)
    model = TinyLM().cuda()
    want_losses, want = _stale_reference(copy.deepcopy(model), 8, lambda ps: torch.optim.AdamW(ps, lr=3e-3), 0)
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3, delay_updates=True).attach_hooks(model)
    fwd = COMPILERS[how](model)
    losses = []
    for k in range(1, 9):
        loss = lm_loss(fwd, k)
        loss.backward()
        opt.step_sync()
        losses.append(loss.item())
    opt.flush()
    assert max(abs(a - b) for a, b in zip(losses, want_losses)) < 1e-3
    assert max((p - w).abs().max().item() for p, w in zip(model.parameters(), want)) < 1e-5
    opt.close()
