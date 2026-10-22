# adamx

AdamW for PyTorch with the optimizer state offloaded to host RAM and updated by
fused SIMD kernels on the CPU. It overlaps those updates with the GPU backward
pass, layer by layer. It implements the *OptiCompute-Bridge* design: the GPU
keeps only the parameters, while FP32 master weights and both Adam moments
live on the (NUMA-local) host.

```python
import adamx

opt = adamx.AdamX(model.parameters(), lr=1e-4, weight_decay=0.01,
                  precision="bf16_moments")   # 'fp32' | 'bf16' | 'fp8'
opt.attach_hooks(model)                       # layer-wise async updates

for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()     # each layer is offloaded + updated as soon as its grad is ready
    opt.step_sync()     # wait; the next forward pass sees the new weights
```

Without `attach_hooks`, `AdamX` is a drop-in optimizer (`backward(); step(); zero_grad()`),
and works with LR schedulers and `clip_grad_norm_`. `adamx.OptiComputeAdamW` is an alias.

## Measured results

The test machine was an RTX 5090 Laptop GPU and a 24-core Intel CPU (AVX2), running Windows.
The model has 806M parameters (24 MLP blocks, d=2048) and each step processes 8192 tokens
(`benchmarks/bench_pipeline.py`):

| configuration                              | step time | peak GPU memory |
|--------------------------------------------|----------:|----------------:|
| forward + backward only (lower bound)      |    497 ms |        9.4 GiB |
| PyTorch fused AdamW on GPU (FP32 + autocast) |  578 ms |       19.2 GiB |
| adamx, synchronous `step()`                |   1180 ms |        9.4 GiB |
| adamx, async hooks, FP32 moments           |    580 ms |        9.4 GiB |
| **adamx, async hooks, BF16 moments**       | **547 ms** |   **9.4 GiB** |

With async hooks, adamx matches or beats the all-GPU optimizer while using half the GPU memory.
The async pipeline is about 2× faster than offloading without overlap.

Host kernel throughput on 64M parameters with 16 threads (`benchmarks/bench_kernels.py`):

| implementation                              | ms/step |
|---------------------------------------------|--------:|
| `torch.optim.AdamW` (foreach, CPU)          |    73.4 |
| `torch.optim.AdamW(fused=True)` (CPU)       |    26.0 |
| adamx native, FP32 moments                  |    26.5 |
| adamx native, BF16 moments + BF16 write-back |   21.3 |
| adamx PyTorch fallback, FP32 / BF16 / FP8   | 40 / 81 / 127 |

The native kernel runs at DRAM bandwidth (~70 GB/s here). Note that recent PyTorch
already ships a fused CPU AdamW at the same speed for FP32 state. adamx's advantage comes
from the overlap pipeline, compressed moments, and the BF16 write-back produced in the same
pass, not from FP32 arithmetic.

## Install

```bash
pip install .               # or: uv pip install .   /   uv add ./adamx
pip install ".[compiler]"   # also installs ziglang, a C compiler for machines without one
python -m adamx build       # optional: compile the native kernels now
python -m adamx info        # shows the ISA, NUMA nodes and L2 size
```

adamx is a pure-Python wheel (`py3-none-any`) that carries its C source, so the same
artifact installs everywhere.

**CUDA torch:** adamx only declares `torch>=2.1`. On Windows, and for specific CUDA
versions, PyPI's `torch` is CPU-only, so a clean `pip`/`uv` install pulls the CPU build.
Install torch from the PyTorch index first (`pip install torch --index-url
https://download.pytorch.org/whl/cu130`). With uv, pin it in your project instead:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cuda" }

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu130"   # pick your CUDA version
explicit = true
```

The native kernel is plain C, loaded through `ctypes`, and compiled on first use into
`~/.cache/adamx` (override with `ADAMX_CACHE_DIR`). The build tries `$ADAMX_CC`, then
`cc`, `gcc`, `clang`, then `python -m ziglang cc` (`pip install ziglang` gives Windows
a compiler), then MSVC `cl`. If none works, adamx falls back to the PyTorch kernels,
which support everything but are slower. Set `ADAMX_DISABLE_NATIVE=1` to force the fallback.

## How it works

```
autograd thread        D2H stream          worker thread (per GPU)        H2D stream
grad of layer L ready
  record event ──────► wait, copy grad
                       to pinned host ───► wait copy, fused AdamW on
                                           NUMA-pinned thread pool ─────► copy weights
backward continues for layer L-1 ...                                      back to GPU
```

`step_sync()` waits for the host work. It then makes the current CUDA stream wait on the
H2D stream, so no device-wide synchronization is needed.

* **Fused kernel** (`adamx/csrc/adamx_kernels.c`): one pass reads p, g, m and v and writes
  p, m, v and the low-precision copy for the GPU. ISA is chosen at runtime: AVX-512F,
  AVX2+FMA, NEON or scalar. The arithmetic is exactly `torch.optim.AdamW`'s.
* **Moment storage** (`precision=`):

  | precision | m | v | host bytes/param | kernel |
  |---|---|---|---:|---|
  | `fp32` | FP32 | FP32 | 12 | native |
  | `bf16` | BF16 | BF16 | 8 | native |
  | `fp8` | E4M3, per-128 scale | E5M2 of `sqrt(v)`, per-128 scale | 6.06 | PyTorch |

* **NUMA**: each GPU's state is allocated by, and updated on, threads pinned to the CPUs
  of the NUMA node that the GPU's PCIe slot hangs off (read from Linux sysfs). This uses
  first-touch placement, so no libnuma is needed. On other platforms pinning is a no-op.
* **Auto-tuning** (`autotune=True` or `AutoTuneConfig(...)`):
  * It benchmarks thread counts for the native kernel.
  * When host update time exceeds GPU backward time, it steps moment precision down a level.
  * It keeps a downgrade only if host time actually falls. Floor: `min_precision="bf16"` by default.

### Where this deviates from the design document, and why

1. **Stochastic rounding for BF16/FP8 moments.** With round-to-nearest and β₂ = 0.999, the
   per-step change in `v` is smaller than half a BF16 ulp, so `v` freezes.
   `test_ema_stalls_without_stochastic_rounding` shows it sticking at its initial value.
   adamx rounds stochastically using a counter-based hash. The native and PyTorch kernels
   use the same hash, so they make the same rounding decisions, and results do not depend
   on the thread split.
2. **FP8 stores `sqrt(v)`, not `v`.** Block-scaled E5M2 `v` flushes to zero for any element
   whose gradient is about 2¹⁶× below its block's maximum, while `m` stays non-zero. The
   step then becomes `m/eps` and explodes. Storing the square root doubles the usable range,
   at the cost of a ~1–2% upward bias in `v`.
3. **BF16 write-back rounds to nearest-even** instead of truncating the top 16 bits.
   Truncation biases every weight toward zero.
4. **The kernel handles tails.** The spec kernel reads and writes past the end when
   `n % 16 != 0`.
5. **Master weights are created lazily** on the first update, not at construction. Loading
   pretrained weights after building the optimizer would otherwise be silently reverted.
6. **SPSC queues are Python `SimpleQueue`s.** There is one producer side (autograd hooks) and
   one consumer thread per device. Lock contention isn't the bottleneck: kernels run with the
   GIL released.

## API notes

* `release_grads=True` (default): in async mode `p.grad` is freed as soon as it is copied
  to the host.
* Gradient accumulation: run all but the last micro-batch under `with opt.no_sync():`.
* Global gradient clipping is impossible in async mode, because the first layers update
  before the last gradients exist. Use synchronous `step()` if you need it.
* `overlap_writeback=False` defers all weight copies to `step_sync()`. You only need it if
  something reads a parameter during backward after that parameter's gradient is complete.
* `state_dict()` saves moments in their stored precision. `load_state_dict()` also accepts
  `torch.optim.AdamW` checkpoints and converts precision when needed.
* `opt.memory_report()`, `opt.last_step_stats`, `opt.autotune_log` and `opt.backend`
  are there for inspection.

## Testing

```bash
pytest                          # 68 tests; CUDA tests run when a GPU is present
cc -O2 -o selftest tests/native/selftest.c adamx/csrc/adamx_kernels.c -lm && ./selftest
```

What has been verified:

* FP32 results equal `torch.optim.AdamW` to about 1e-7, on CPU and CUDA, in sync and async modes.
* Covered: parameter groups, schedulers, weight tying, gradient accumulation, BF16 params,
  exact checkpoint resume, and the auto-tuner.
* The C self-test checks every ISA path against scalar code under ASan/UBSan.
* The suite runs on Windows (zig) and Linux (gcc).

## Limitations / not yet verified

* The **AVX-512 and NEON paths compile but were not executed**: the test machine is AVX2.
  Run `selftest` on such hardware before relying on them.
* **Multi-socket NUMA placement** is covered by unit tests on a simulated sysfs, not
  measured on a real two-socket server.
* FP8 moments run on the PyTorch kernel, which is slower than native BF16. The auto-tuner
  won't move to FP8 unless it's actually faster, and FP8 is opt-in via `min_precision`.
* Not covered yet: FP16 gradients with `GradScaler`, and integration with DDP or FSDP
  sharding (single-process, one or more GPUs).
#   a d a m x  
 #   a d a m x  
 