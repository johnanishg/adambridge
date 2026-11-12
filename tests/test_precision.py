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


def test_master_precision_aliases_and_bytes():
    assert P.canonical_master_precision("FP24") == "fp24"
    assert P.canonical_master_precision("float32") == "fp32"
    with pytest.raises(ValueError):
        P.canonical_master_precision("fp16")
    assert P.bytes_per_param("fp8", master="fp24") == pytest.approx(5.0625)
    assert P.bytes_per_param("fp32", moments=1, master="fp24") == 7


def test_fp24_nearest_even_matches_brute_force():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(100_000, generator=g) * torch.exp2(torch.randint(-20, 20, (100_000,), generator=g).float())
    b = x.view(torch.int32)
    down = b & ~0xFF
    up = down + 0x100
    dd = (x.double() - down.view(torch.float32).double()).abs()
    du = (up.view(torch.float32).double() - x.double()).abs()
    want = torch.where(dd < du, down, torch.where(du < dd, up, torch.where((down & 0x100) != 0, up, down)))
    got = P.round_to_fp24_nearest_(x.clone()).view(torch.int32)
    assert torch.equal(got, want)
    nan = P.round_to_fp24_nearest_(torch.tensor([float("nan"), 1.0, -float("inf")]))
    assert nan[0].isnan() and nan[1] == 1.0 and nan[2] == -float("inf")


def test_fp24_stochastic_rounding_is_unbiased():
    x = torch.full((200_000,), 1.0 + 2**-17)  # a quarter of the way between two 24-bit values
    y = P.round_to_fp24_stochastic(x.clone(), seed=0)
    assert set(y.unique().tolist()) == {1.0, 1.0 + 2**-15}
    assert y.mean().item() == pytest.approx(1.0 + 2**-17, abs=0.01 * 2**-15)


def test_fp24_store_round_trips_and_converts():
    n = 1000
    x = torch.randn(n) * 3
    store = P.make_master("fp24", n, init=x)
    y = store.to_fp32()
    assert store.nbytes == 3 * n
    assert ((y - x).abs() <= x.abs() * 2**-16).all()           # nearest even: within half an ulp
    assert (y.view(torch.int32) & 0xFF == 0).all()              # the low byte is gone
    assert torch.equal(P.make_master("fp24", n, init=y).to_fp32(), y)  # 24-bit values are exact
    clone = P.FP24Store(n)
    P.load_master_(clone, store.state_dict())
    assert torch.equal(clone.to_fp32(), y)
    f32 = P.make_master("fp32", n)
    P.load_master_(f32, store.state_dict())                     # fp24 -> fp32: exact
    assert torch.equal(f32.to_fp32(), y)
    back = P.make_master("fp24", n)
    P.load_master_(back, f32.data)                              # fp32 tensor -> fp24: exact here, and no clobbering
    assert torch.equal(back.to_fp32(), y) and torch.equal(f32.data, y)
    for bad in ({"precision": "fp24", "hi": torch.zeros(1, dtype=torch.bfloat16), "lo": torch.zeros(n, dtype=torch.uint8)},
                {"precision": "fp24", "hi": torch.zeros(n, dtype=torch.int16), "lo": torch.zeros(n, dtype=torch.uint8)},
                {"precision": "fp24", "hi": torch.zeros(n, dtype=torch.bfloat16), "lo": torch.zeros(n)},
                {"precision": "fp16", "data": torch.zeros(n)}, "fp24", torch.zeros(n - 1)):
        with pytest.raises(ValueError):
            P.check_master_state(bad, n, "master")


def test_fp24_store_initialises_exactly_from_bf16_and_in_chunks(monkeypatch):
    n = 1000
    x = torch.randn(n) * 3
    from_bf16 = P.make_master("fp24", n, init=x.to(torch.bfloat16).view(10, 100))
    assert torch.equal(from_bf16.to_fp32(), x.to(torch.bfloat16).float()) and not from_bf16.lo.any()
    whole = P.make_master("fp24", n, init=x)
    monkeypatch.setattr(P.FP24Store, "INIT_CHUNK", 96)
    chunked = P.make_master("fp24", n, init=x.view(20, 50))
    assert torch.equal(chunked.to_fp32(), whole.to_fp32())


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
