"""End-to-end training step: GPU AdamW vs adamx (synchronous) vs adamx (layer-wise async).

    python benchmarks/bench_pipeline.py --dim 2048 --layers 24 --tokens 8192

Reports median step time and peak GPU memory. "fwd+bwd only" is the floor
any offloading scheme can approach; the async pipeline's goal is to hide the
host update behind the backward pass.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time

import torch

import adamx


class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.up = torch.nn.Linear(dim, 4 * dim)
        self.down = torch.nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


def make_model(dim, layers, dtype):
    torch.manual_seed(0)
    return torch.nn.Sequential(*[Block(dim) for _ in range(layers)]).cuda().to(dtype)


def run(name, model, step_fn, x, steps, warmup):
    times = []
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step_fn(model, x)
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30
    med = statistics.median(times)
    print(f"{name:<44}{med * 1e3:>9.1f} ms{peak:>9.2f} GiB")
    return med


def fwd_bwd(model, x):
    model(x).float().pow(2).mean().backward()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--precision", default="fp32", help="adamx moment precision")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    n = sum(p.numel() for p in make_model(args.dim, 1, torch.float32).parameters()) * args.layers
    gc.collect()
    print(f"{n / 1e6:.0f}M params, {args.tokens} tokens/step, adamx moments={args.precision}")
    print(f"{'configuration':<44}{'step':>12}{'peak GPU':>13}")

    # Floor: forward + backward in BF16, no optimizer.
    model = make_model(args.dim, args.layers, torch.bfloat16)
    x = torch.randn(args.tokens, args.dim, device="cuda", dtype=torch.bfloat16)

    def floor(m, x):
        fwd_bwd(m, x)
        for p in m.parameters():
            p.grad = None

    run("fwd+bwd only (bf16, no optimizer)", model, floor, x, args.steps, args.warmup)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Baseline: everything on the GPU, FP32 params + fused AdamW, BF16 autocast.
    model = make_model(args.dim, args.layers, torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    def gpu_step(m, x, opt=opt):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fwd_bwd(m, x)
        opt.step()
        opt.zero_grad(set_to_none=True)

    try:
        run("GPU fused AdamW (fp32 params + autocast)", model, gpu_step, x, args.steps, args.warmup)
    except torch.OutOfMemoryError:
        print(f"{'GPU fused AdamW (fp32 params + autocast)':<44}{'OOM':>12}")
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    for mode in ("sync", "async"):
        model = make_model(args.dim, args.layers, torch.bfloat16)
        opt = adamx.AdamX(model.parameters(), lr=1e-4, precision=args.precision, num_threads=args.threads)
        if mode == "async":
            opt.attach_hooks(model)

            def step(m, x, opt=opt):
                fwd_bwd(m, x)
                opt.step_sync()
        else:

            def step(m, x, opt=opt):
                fwd_bwd(m, x)
                opt.step()
                opt.zero_grad(set_to_none=True)

        label = f"adamx {mode:<5} ({opt.backend}, bf16 params)"
        run(label, model, step, x, args.steps, args.warmup)
        s = opt.last_step_stats
        print(f"{'':<4}host kernel time {s.cpu_update * 1e3:.0f} ms, step_sync stall {s.stall * 1e3:.0f} ms, "
              f"host state {opt.memory_report()['total'] / 2**30:.1f} GiB")
        opt.close()
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
