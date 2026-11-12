import pytest
import torch

from adambridge import native
from adambridge.engine import UpdateEngine
from adambridge.kernels import StepCoefficients, fused_adamw_chunk, fused_lion_chunk, split_range
from adambridge.precision import make_master, make_store

from conftest import NATIVE

needs_native = pytest.mark.skipif(not NATIVE, reason="native kernels unavailable")

N = 10_007  # odd, exercises SIMD tails


def make_inputs(prec, grad_dtype, out_dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    master = torch.randn(N, generator=g)
    grad = (torch.randn(N, generator=g) * 1e-2).to(grad_dtype)
    m = make_store(prec, N, "m", init=torch.randn(N, generator=g) * 1e-3)
    v = make_store(prec, N, "v", init=torch.rand(N, generator=g) * 1e-4)
    out = None if out_dtype is None else torch.empty(N, dtype=out_dtype)
    return master, grad, m, v, out


def test_split_range():
    assert split_range(0, 4, 128) == []
    assert split_range(100, 4, 128) == [(0, 100)]
    ranges = split_range(1000, 3, 128)
    assert ranges[0][0] == 0 and ranges[-1][1] == 1000
    assert all(lo % 128 == 0 for lo, _ in ranges)
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_torch_kernel_matches_torch_adamw():
    master, grad, m, v, _ = make_inputs("fp32", torch.float32, None)
    p = torch.nn.Parameter(master.clone())
    ref = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.1, foreach=False)
    ref.state[p] = {"step": torch.tensor(4.0), "exp_avg": m.to_fp32(), "exp_avg_sq": v.to_fp32()}
    p.grad = grad.clone()
    ref.step()
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.1, 5)
    fused_adamw_chunk(master, grad, m, v, None, c, 0, N)
    torch.testing.assert_close(master, p.detach(), rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(m.to_fp32(), ref.state[p]["exp_avg"], rtol=1e-6, atol=1e-9)


@needs_native
@pytest.mark.parametrize("prec", ["fp32", "bf16"])
@pytest.mark.parametrize("grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32, torch.bfloat16])
def test_native_matches_torch_kernel(prec, grad_dtype, out_dtype):
    kernels = native.load()
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 3)
    a = make_inputs(prec, grad_dtype, out_dtype)
    b = make_inputs(prec, grad_dtype, out_dtype)
    kernels.fused_adamw(a[0], a[1], a[2].raw, a[3].raw, a[4], kernels.make_hparams(c, 0, False), 0, N)
    fused_adamw_chunk(*b, c, 0, N)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
    # BF16 storage: an FMA-vs-separate rounding difference may tip one ulp.
    tol = dict(rtol=1e-5, atol=1e-9) if prec == "fp32" else dict(rtol=2**-7, atol=1e-9)
    torch.testing.assert_close(a[2].to_fp32(), b[2].to_fp32(), **tol)
    torch.testing.assert_close(a[3].to_fp32(), b[3].to_fp32(), **tol)
    if out_dtype is not None:
        torch.testing.assert_close(a[4].float(), b[4].float(), rtol=2**-7 if out_dtype == torch.bfloat16 else 1e-5, atol=1e-6)
        if out_dtype == torch.bfloat16:  # round-to-nearest-even, identical to torch's cast
            assert torch.equal(a[4], a[0].to(torch.bfloat16))


@needs_native
def test_native_simd_matches_scalar():
    kernels = native.load()
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 7)
    a = make_inputs("bf16", torch.bfloat16, torch.bfloat16)
    b = make_inputs("bf16", torch.bfloat16, torch.bfloat16)
    hp = kernels.make_hparams(c, 1234, True)
    try:
        assert kernels.set_simd_level("scalar") == "scalar"
        kernels.fused_adamw(a[0], a[1], a[2].raw, a[3].raw, a[4], hp, 0, N)
    finally:
        kernels.set_simd_level(None)
    kernels.fused_adamw(b[0], b[1], b[2].raw, b[3].raw, b[4], hp, 0, N)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
    # Same hash-based random stream in every ISA: SR decisions agree except
    # where FMA moved the pre-rounding value across a boundary.
    agree = (a[2].raw == b[2].raw).float().mean().item()
    assert agree > 0.99


@needs_native
def test_native_and_torch_make_the_same_rounding_decisions():
    """Both kernels draw stochastic-rounding bits from lowbias32(seed, index)."""
    kernels = native.load()
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 7)
    a = make_inputs("bf16", torch.float32, None)
    b = make_inputs("bf16", torch.float32, None)
    kernels.fused_adamw(a[0], a[1], a[2].raw, a[3].raw, None, kernels.make_hparams(c, 4321, True), 0, N)
    fused_adamw_chunk(*b, c, 0, N, seed=4321)
    for x, y in ((a[2].raw, b[2].raw), (a[3].raw, b[3].raw)):
        assert (x == y).float().mean().item() > 0.99


@needs_native
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("out_dtype", [None, torch.bfloat16])
def test_native_fp8_matches_torch_fp8(stochastic, out_dtype):
    """The native FP8 kernel and the PyTorch FP8 path share layout, arithmetic
    and the stochastic-rounding stream: same weights, (almost) the same codes."""
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 3)
    results = []
    for backend in ("native", "torch"):
        eng = UpdateEngine(precision="fp8", backend=backend, numa_aware=False, stochastic_rounding=stochastic)
        try:
            pool = eng.ensure_pool(torch.device("cpu"))
            master, grad, m, v, out = make_inputs("fp8", torch.bfloat16, out_dtype)
            for _ in range(3):  # several steps, so block scales evolve
                eng.run_kernel(master, grad, m, v, out, c, 1234, pool)
            results.append((master, m, v, out))
        finally:
            eng.close()
    (pa, ma, va, oa), (pb, mb, vb, ob) = results
    torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)
    # FMA vs separate multiply-add can move a pre-rounding value across a
    # rounding boundary; everything else must agree exactly.
    assert (ma.q.view(torch.uint8) == mb.q.view(torch.uint8)).float().mean().item() > 0.99
    assert (va.q.view(torch.uint8) == vb.q.view(torch.uint8)).float().mean().item() > 0.99
    torch.testing.assert_close(ma.scale, mb.scale, rtol=1e-5, atol=0)
    torch.testing.assert_close(va.scale, vb.scale, rtol=1e-5, atol=0)
    if out_dtype is not None:
        assert torch.equal(oa, pa.to(out_dtype))


@pytest.mark.parametrize("backend", ["torch"] + (["native"] if NATIVE else []))
@pytest.mark.parametrize("master_prec", ["fp32", "fp24"])
@pytest.mark.parametrize("prec", ["fp32", "bf16", "fp8"])
def test_results_independent_of_thread_count(backend, prec, master_prec):
    """Including stochastic rounding: RNG streams are keyed by element/chunk, not thread."""
    c = StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 2)
    results = []
    for threads in (1, 3, 8):
        eng = UpdateEngine(precision=prec, backend=backend, numa_aware=False, num_threads=threads, chunk_size=1024)
        try:
            pool = eng.ensure_pool(torch.device("cpu"))
            master, grad, m, v, out = make_inputs(prec, torch.bfloat16, torch.bfloat16)
            if master_prec == "fp24":
                master = make_master("fp24", N, init=master)
            eng.run_kernel(master, grad, m, v, out, c, 99, pool)
            results.append((master if isinstance(master, torch.Tensor) else master.to_fp32(),
                            m.to_fp32(), v.to_fp32(), out))
        finally:
            eng.close()
    for other in results[1:]:
        for x, y in zip(results[0], other):
            assert torch.equal(x, y)


def _fp24_inputs(prec, lion=False):
    master, grad, m, v, out = make_inputs(prec, torch.bfloat16, torch.bfloat16)
    return make_master("fp24", N, init=master), grad, m, None if lion else v, out


@pytest.mark.parametrize("lion", [False, True])
def test_torch_kernel_fp24_master_tracks_fp32_and_writes_the_stored_value(lion):
    """The PyTorch kernel with an FP24 master: within one 24-bit ulp of the FP32 result, and ``out`` is the
    stored (rounded) master cast to BF16."""
    c = (StepCoefficients.for_lion(1e-3, 0.9, 0.99, 0.01) if lion
         else StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 3))
    f32 = make_inputs("bf16", torch.bfloat16, torch.bfloat16)
    f24 = _fp24_inputs("bf16", lion)
    for seed in (None, 4321):
        if lion:
            fused_lion_chunk(f32[0], f32[1], f32[2], f32[4], c, 0, N, seed=seed)
            fused_lion_chunk(f24[0], f24[1], f24[2], f24[4], c, 0, N, seed=seed)
        else:
            fused_adamw_chunk(*f32, c, 0, N, seed=seed)
            fused_adamw_chunk(*f24, c, 0, N, seed=seed)
        stored = f24[0].to_fp32()
        assert ((stored - f32[0]).abs() <= f32[0].abs() * 2**-13).all()
        assert torch.equal(f24[4], stored.to(torch.bfloat16))


@needs_native
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("prec", ["fp32", "fp8"])
@pytest.mark.parametrize("lion", [False, True])
def test_native_fp24_master_matches_torch(stochastic, prec, lion):
    """Native and PyTorch kernels with an FP24 master: the same stored bits (bar FMA boundary cases), and the
    accelerator copy is always the stored value rounded."""
    c = (StepCoefficients.for_lion(1e-3, 0.9, 0.99, 0.01) if lion
         else StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.01, 3))
    results = []
    for backend in ("native", "torch"):
        eng = UpdateEngine(precision=prec, backend=backend, numa_aware=False, stochastic_rounding=stochastic,
                           algorithm="lion" if lion else "adamw")
        try:
            pool = eng.ensure_pool(torch.device("cpu"))
            master, grad, m, v, out = _fp24_inputs(prec, lion)
            for _ in range(3):
                eng.run_kernel(master, grad, m, v, out, c, 1234, pool)
            results.append((master, out))
        finally:
            eng.close()
    (sa, oa), (sb, ob) = results
    pa, pb = sa.to_fp32(), sb.to_fp32()
    torch.testing.assert_close(pa, pb, rtol=2**-12, atol=1e-6)
    same = ((sa.hi == sb.hi) & (sa.lo == sb.lo)).float().mean().item()
    assert same > 0.99
    assert torch.equal(oa, pa.to(torch.bfloat16)) and torch.equal(ob, pb.to(torch.bfloat16))
