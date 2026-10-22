"""Host AdamW step throughput: adamx kernels vs torch.optim.AdamW on CPU.

    python benchmarks/bench_kernels.py --numel 64M --threads 8

Reports wall time per step and effective DRAM bandwidth (bytes each
implementation must move at minimum: read p,g,m,v; write p,m,v[,out]).
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from adamx import native
from adamx.engine import UpdateEngine
from adamx.kernels import StepCoefficients
from adamx.precision import make_store


def parse_numel(text: str) -> int:
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    return int(float(text[:-1]) * mult[text[-1].upper()]) if text[-1].upper() in mult else int(text)


def timeit(fn, repeats):
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_torch_adamw(n, repeats, **kw):
    p = torch.nn.Parameter(torch.randn(n))
    p.grad = torch.randn(n) * 1e-3
    opt = torch.optim.AdamW([p], lr=1e-4, **kw)
    return timeit(opt.step, repeats)


def bench_adamx(n, repeats, precision, backend, threads, grad_dtype, out_dtype):
    eng = UpdateEngine(precision=precision, backend=backend, numa_aware=False, num_threads=threads)
    try:
        pool = eng.ensure_pool(torch.device("cpu"))
        master = torch.randn(n)
        grad = (torch.randn(n) * 1e-3).to(grad_dtype)
        m, v = make_store(precision, n, "m"), make_store(precision, n, "v")
        out = None if out_dtype is None else torch.empty(n, dtype=out_dtype)
        c = StepCoefficients.for_step(1e-4, 0.9, 0.999, 1e-8, 0.01, 10)
        return timeit(lambda: eng.run_kernel(master, grad, m, v, out, c, 0, pool), repeats)
    finally:
        eng.close()


def min_bytes(precision, grad_bytes, out_bytes):
    mv = {"fp32": 4, "bf16": 2, "fp8": 1 + 4 / 128}[precision]
    return 4 * 2 + grad_bytes + mv * 4 + out_bytes  # p r/w, g r, m+v r/w, out w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numel", default="32M")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    n = parse_numel(args.numel)
    threads = args.threads or torch.get_num_threads()
    torch.set_num_threads(threads)
    kernels = native.load()
    print(f"numel={n:,}  threads={threads}  native={kernels.simd_level if kernels else 'unavailable'}")
    print(f"{'implementation':<52}{'ms/step':>10}{'GB/s':>8}")

    def row(name, seconds, nbytes):
        print(f"{name:<52}{seconds * 1e3:>10.1f}{nbytes * n / seconds / 1e9:>8.1f}")

    row("torch.optim.AdamW (foreach)", bench_torch_adamw(n, args.repeats, foreach=True), min_bytes("fp32", 4, 0))
    try:
        row("torch.optim.AdamW (fused=True)", bench_torch_adamw(n, args.repeats, fused=True), min_bytes("fp32", 4, 0))
    except (RuntimeError, TypeError) as exc:
        print(f"torch.optim.AdamW (fused=True): unavailable ({exc})")

    configs = [
        ("fp32", torch.float32, None),
        ("fp32", torch.bfloat16, torch.bfloat16),
        ("bf16", torch.bfloat16, torch.bfloat16),
        ("fp8", torch.bfloat16, torch.bfloat16),
    ]
    for backend in (["native"] if kernels else []) + ["torch"]:
        for precision, gdt, odt in configs:
            if backend == "native" and precision == "fp8":
                continue  # FP8 always runs on the PyTorch kernel
            label = f"adamx {backend:<6} moments={precision:<4} grad={str(gdt)[6:]:<8} out={str(odt)[6:] if odt else '-'}"
            t = bench_adamx(n, args.repeats, precision, backend, threads, gdt, odt)
            row(label, t, min_bytes(precision, 2 if gdt == torch.bfloat16 else 4, 2 if odt else 0))


if __name__ == "__main__":
    main()
