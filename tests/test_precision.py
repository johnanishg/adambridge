import pytest
import torch

from adambridge import precision as P
from adambridge.kernels import mix_seed


def test_aliases_and_bytes():
    assert P.canonical_precision("bf16_moments") == "bf16"
    assert P.canonical_precision("FP8_MOMENTS") == "fp8"
    with pytest.raises(ValueError):
        P.canonical_precision("int4")
    assert P.bytes_per_param("fp32") == 12
    assert P.bytes_per_param("bf16") == 8
    assert P.bytes_per_param("fp8") == pytest.approx(6.0625)


def test_bf16_stochastic_rounding_is_unbiased():
    x = torch.full((200_000,), 1.0 + 2**-10)  # 1/8 of the way between two bf16 values
    y = P.round_to_bf16_stochastic(x.clone(), seed=0).float()
    assert set(y.unique().tolist()) == {1.0, 1.0078125}
    assert y.mean().item() == pytest.approx(1.0 + 2**-10, rel=2e-5)


def test_fp8_stochastic_rounding_is_unbiased():
    x = torch.full((200_000,), 1.03)
    y = P.round_to_fp8_stochastic(x.clone(), torch.float8_e4m3fn, seed=0).float()
    assert y.mean().item() == pytest.approx(1.03, rel=1e-3)


@pytest.mark.parametrize("prec", ["bf16", "fp8"])
def test_ema_stalls_without_stochastic_rounding(prec):
    """The reason SR is on by default: v = 0.999 v + 0.001 g^2 does not move
    under round-to-nearest when g^2 is close to v."""
    n = 4096
    # Each 128-block holds one larger, stationary element (as real blocks do),
    # so FP8 block scaling cannot place every element exactly on the grid.
    init = torch.ones(n)
    init[::128] = 4.0
    target = torch.full((n,), 1.5)
    target[::128] = 4.0
    tracked = torch.ones(n, dtype=torch.bool)
    tracked[::128] = False
    results = {}
    for sr in (False, True):
        store = P.make_store(prec, n, "v", init=init)
        for step in range(500):
            v = store.read(0, n)
            v.mul_(0.999).add_(target, alpha=0.001)
            store.write(0, n, v, seed=P.moment_seed(mix_seed(0, step), "v") if sr else None)
        results[sr] = store.to_fp32()[tracked].mean().item()
    exact = 1.5 + (1.0 - 1.5) * 0.999**500  # ~1.197
    assert results[False] == pytest.approx(1.0, abs=1e-6)  # stuck at the initial value
    # FP8 keeps sqrt(v): rounding is unbiased for sqrt(v), so v carries a small upward bias.
    assert results[True] == pytest.approx(exact, rel=0.01 if prec == "bf16" else 0.03)


def test_fp8_store_keeps_small_second_moments():
    """sqrt-domain storage keeps v non-zero for gradients 2^20 x below the block max."""
    v = torch.full((128,), 1e-12)
    v[0] = 1.0
    store = P.make_store("fp8", 128, "v", init=v)
    out = store.to_fp32()
    assert out[0].item() == pytest.approx(1.0, rel=0.3)
    assert (out[1:] > 0).all()


@pytest.mark.parametrize("prec", ["fp32", "bf16", "fp8"])
@pytest.mark.parametrize("role", ["m", "v"])
def test_roundtrip_and_state_dict(prec, role):
    n = 1000  # not a multiple of the FP8 block
    x = torch.randn(n) * 1e-3
    if role == "v":
        x = x.abs()
    store = P.make_store(prec, n, role, init=x)
    tol = {"fp32": 0, "bf16": 2**-8, "fp8": 0.25}[prec]
    err = (store.to_fp32() - x).abs().max().item() / x.abs().max().item()
    assert err <= tol
    clone = P.store_from_state_dict(store.state_dict(), n, role)
    assert torch.equal(clone.to_fp32(), store.to_fp32())


def test_fp8_chunked_writes_match_whole():
    n = 1000
    x = torch.randn(n)
    whole = P.make_store("fp8", n, "m")
    whole.write(0, n, x.clone())
    parts = P.make_store("fp8", n, "m")
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        parts.write(lo, hi, x[lo:hi].clone())
    assert torch.equal(whole.to_fp32(), parts.to_fp32())
