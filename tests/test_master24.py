"""master_precision='fp24': the optimizer trains like one with an FP32 master, saves and loads in every direction,
and holds 3 bytes per parameter of master weights."""

import pytest
import torch

import adambridge
from conftest import max_param_diff, tiny_mlp

LR = 1e-2


def _model(seed=0, device="cpu"):
    # BF16 parameters: the master is a separate buffer, so the FP24 format applies (an FP32 CPU parameter is
    # its own master and stays FP32 whatever master_precision says).
    return tiny_mlp(seed, dtype=torch.bfloat16, device=device)


def _train(pairs, steps, device="cpu", hooks=False):
    """Every optimizer gets the same gradients each step (through autograd, so the hooks fire), so the only
    difference between the models is the master precision: a real loss would let Adam amplify one BF16 ulp of
    weight difference into different updates wherever the gradient is tiny."""
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        grads = [(torch.randn(p.shape, generator=g) * 0.1).to(device=device, dtype=torch.bfloat16)
                 for p in pairs[0][0].parameters()]
        for model, opt in pairs:
            sum((p * grad).sum() for p, grad in zip(model.parameters(), grads)).backward()
            if hooks:
                opt.step_sync()
            else:
                opt.step()
                opt.zero_grad(set_to_none=True)


def _opt(model, **kw):
    return adambridge.AdamBridge(model.parameters(), lr=LR, weight_decay=0.0, small_param_threshold=0, **kw)


@pytest.mark.parametrize("precision", ["fp32", "fp8"])
def test_fp24_master_tracks_the_fp32_master(precision):
    a, b = _model(), _model()
    opt_a, opt_b = _opt(a, precision=precision), _opt(b, precision=precision, master_precision="fp24")
    assert opt_b.expected_bytes_per_param() == opt_a.expected_bytes_per_param() - 1
    _train([(a, opt_a), (b, opt_b)], 20)
    for ma, mb in zip(opt_a.master_params(), opt_b.master_params()):
        assert ((ma - mb).abs() <= ma.abs() * 1e-3 + 1e-5).all()
    assert max_param_diff(a, b) <= 2**-7  # BF16 parameters: within one BF16 ulp of each other
    report = opt_b.memory_report()
    n = sum(p.numel() for p in b.parameters())
    assert report["master"] == 3 * n
    assert opt_a.memory_report()["master"] == 4 * n
    opt_a.close()
    opt_b.close()


def test_fp32_cpu_parameters_stay_their_own_master():
    model = tiny_mlp(0)
    opt = _opt(model, master_precision="fp24")
    _train([(model, opt)], 2)
    assert opt.memory_report()["master"] == 0
    assert all(m.dtype == torch.float32 and m.data_ptr() == p.data_ptr()
               for p, m in zip(model.parameters(), opt.master_params()))
    opt.close()


def test_fp24_checkpoints_round_trip_and_convert():
    model = _model()
    opt = _opt(model, precision="bf16", master_precision="fp24")
    _train([(model, opt)], 3)
    sd = opt.state_dict()
    assert sd["adambridge"]["master_precision"] == "fp24"
    saved = sd["state"][0]["master"]
    assert saved["precision"] == "fp24" and saved["hi"].dtype == torch.bfloat16 and saved["lo"].dtype == torch.uint8
    masters = [m.clone() for m in opt.master_params()]

    def loaded(master_precision, state):
        other = _opt(_model(1), precision="bf16", master_precision=master_precision)
        other.load_state_dict(state)
        return other

    same = loaded("fp24", sd)                       # fp24 -> fp24: the planes are copied
    as32 = loaded("fp32", sd)                       # fp24 -> fp32: exact (24 bits fit)
    back = loaded("fp24", as32.state_dict())        # fp32 tensors -> fp24: exact for 24-bit values
    for other in (same, as32, back):
        for x, y in zip(masters, other.master_params()):
            assert torch.equal(x, y)
        assert other._ordered[0].step == 3
        other.close()
    plain = {**sd, "state": {k: {**v, "master": masters[k]} for k, v in sd["state"].items()}}
    from_tensors = loaded("fp24", plain)            # plain FP32 tensors (an AdamW-style layout) load too
    for x, y in zip(masters, from_tensors.master_params()):
        assert torch.equal(x, y)
    from_tensors.close()
    opt.close()


def test_bad_fp24_checkpoints_are_rejected_whole():
    model = _model()
    opt = _opt(model, master_precision="fp24")
    _train([(model, opt)], 1)
    sd = opt.state_dict()
    n = sd["state"][0]["master"]["hi"].numel()
    for bad_master in ({"precision": "fp24", "hi": torch.zeros(1, dtype=torch.bfloat16), "lo": torch.zeros(n, dtype=torch.uint8)},
                       {"precision": "fp24", "hi": torch.zeros(n, dtype=torch.bfloat16), "lo": torch.zeros(n, dtype=torch.int8)},
                       {"precision": "fp16", "hi": torch.zeros(n, dtype=torch.bfloat16)}):
        bad = {**sd, "state": {**sd["state"], 0: {**sd["state"][0], "master": bad_master}}}
        other = _opt(_model(1), master_precision="fp24")
        with pytest.raises(ValueError):
            other.load_state_dict(bad)
        other.load_state_dict(sd)  # nothing was half-applied
        other.close()
    opt.close()


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["fp32", "fp8"])
def test_fp24_master_on_cuda_with_hooks(precision):
    a, b = _model(device="cuda"), _model(device="cuda")
    opt_a = _opt(a, precision=precision).attach_hooks(a)
    opt_b = _opt(b, precision=precision, master_precision="fp24").attach_hooks(b)
    _train([(a, opt_a), (b, opt_b)], 20, device="cuda", hooks=True)
    for ma, mb in zip(opt_a.master_params(), opt_b.master_params()):
        assert ((ma - mb).abs() <= ma.abs() * 1e-3 + 1e-5).all()
    assert max_param_diff(a, b) <= 2**-7
    n = sum(p.numel() for p in b.parameters())
    assert opt_b.memory_report()["master"] == 3 * n
    opt_a.close()
    opt_b.close()


@pytest.mark.cuda
def test_fp24_slots_moved_to_the_gpu_become_fp32():
    model = _model(device="cuda")
    opt = _opt(model, master_precision="fp24", gpu_state_budget=1 << 40).attach_hooks(model)
    _train([(model, opt)], 3, device="cuda", hooks=True)
    assert all(s.gpu is not None for s in opt._ordered)
    sd = opt.state_dict()
    assert all(isinstance(st["master"], torch.Tensor) for st in sd["state"].values())
    other = _opt(_model(1, device="cuda"), master_precision="fp24")
    other.load_state_dict(sd)  # FP32 tensors into FP24 host masters
    for x, y in zip(opt.master_params(), other.master_params()):
        assert ((x.reshape(-1).cpu() - y).abs() <= x.reshape(-1).abs().cpu() * 2**-16).all()
    other.close()
    opt.close()
