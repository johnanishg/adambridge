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

The test machine is a laptop: RTX 5090 Laptop GPU (24 GB), 24-core Intel CPU (AVX2),
64 GB RAM, Windows.

### Fine-tuning a real LLM

Qwen2.5-0.5B (494M parameters) fine-tuned on WikiText-2: 200 steps of 8 × 512 tokens,
peak learning rate 2e-5 with warm-up and cosine decay. Every configuration starts from
the same weights, sees the same batches in the same order, and runs in a fresh process
(the *LLM fine-tuning* experiment in `adamx-lab`). Validation loss is measured on the
whole WikiText-2 validation split (start: 2.868).

| configuration | final val. loss | Δ vs reference | step | peak GPU | host RAM† |
|---|---:|---:|---:|---:|---:|
| PyTorch AdamW, FP32 weights + autocast (**reference**) | 2.6402 | | 292 ms | 19.6 GiB | |
| PyTorch AdamW, BF16 weights + FP32 master, all on GPU | 2.6404 | +0.0002 | 281 ms | 18.8 GiB | |
| **adamx, FP32 moments** | 2.6409 | +0.0008 | 395 ms | 13.2 GiB | +7.4 GiB |
| **adamx, BF16 moments** | 2.6405 | +0.0003 | 317 ms | 13.2 GiB | +5.6 GiB |
| **adamx, FP8 moments** | 2.6409 | +0.0007 | 305 ms | 13.2 GiB | +4.7 GiB |
| bitsandbytes AdamW8bit, FP32 weights | 2.6433 | +0.0031 | 312 ms | 16.8 GiB | |
| bitsandbytes AdamW8bit, BF16 weights | 2.6557 | +0.0156 | 298 ms | 14.2 GiB | |
| bitsandbytes PagedAdamW8bit, BF16 weights | 2.6549 | +0.0147 | 306 ms | 13.3 GiB | |
| torchao CPUOffloadOptimizer, FP32 weights | 2.6401 | −0.0001 | 525 ms | 15.9 GiB | +10.6 GiB |
| torchao CPUOffloadOptimizer, BF16 weights | 2.6526 | +0.0124 | 341 ms | 13.2 GiB | +5.3 GiB |
| PyTorch AdamW, BF16 weights only | 2.6526 | +0.0124 | 266 ms | 15.1 GiB | |

† growth of the process's resident memory beyond the all-GPU reference's.

* **Noise.** Re-running the reference with two other data orders moved its final loss by
  up to 0.0017, so differences below 0.002 are noise. All three adamx precisions are
  inside that band, and so is the same algorithm run entirely on the GPU: offloading the
  optimizer costs no quality.
* **Why the BF16-only configurations fall behind.** An AdamW update at this learning rate is
  smaller than half the BF16 spacing of most weights, so without an FP32 copy it is rounded
  away: only 18% of their weights changed (at BF16 resolution) vs 84% for the reference.
  adamx keeps BF16 weights on the GPU but the FP32 master on the host.
* **Memory.** adamx needs 32% less GPU memory than the reference, the least of any
  configuration at reference quality, and 2.7 GiB less than torchao's offload at the same quality.
* **Speed.** With BF16 or FP8 moments adamx is 5–9% slower than the all-GPU reference and
  needs 40% less time per step than torchao's offload at the same quality (25% less with FP32 moments). The GPU waits once per step for the
  last gradient's update: the embedding (136M parameters), which backward produces last and the
  next forward needs first. With FP32 moments that update moves 28 bytes per parameter
  through host memory, hence 395 ms. This laptop's GPU also drops its clocks whenever it
  idles: a forward+backward pass that takes 251 ms back to back takes 22% longer after a 20 ms
  pause and 33% longer after a 50 ms one, which every CPU-offload optimizer pays.
* **Host memory.** adamx's host footprint is its FP32 master, the moments and one gradient and
  one write-back buffer per parameter: with FP8 moments, less than torchao's BF16 offload,
  which keeps no FP32 master.

### Host kernel

32M parameters, 24 threads (`benchmarks/bench_kernels.py --numel 32M`):

| implementation | ms/step | GB/s |
|---|---:|---:|
| `torch.optim.AdamW` (foreach, CPU) | 31.9 | 28 |
| `torch.optim.AdamW(fused=True)` (CPU) | 11.2 | 80 |
| adamx native, FP32 moments | 10.5 | 85 |
| adamx native, FP32 moments, BF16 gradient in + BF16 weights out | 11.8 | 76 |
| adamx native, BF16 moments (same I/O) | 8.3 | 77 |
| adamx native, FP8 moments (same I/O) | 7.2 | 72 |
| adamx PyTorch fallback, FP32 / BF16 / FP8 moments | 16.0 / 32.0 / 45.2 | |

The native kernel runs at DRAM bandwidth. Recent PyTorch already ships a fused CPU AdamW at
the same speed for FP32 state; adamx's advantage comes from the overlap pipeline, compressed
moments, and the BF16 write-back produced in the same pass, not from FP32 arithmetic.

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

* **Scheduling.** Autograd's CPU side runs ahead of the GPU, so a parameter's hook fires
  long before the GPU has produced its gradient. The worker updates, in one native call,
  every gradient whose copy has already landed (copies complete in order on one stream),
  so the host update stays right behind the GPU's backward pass. Under load the batches
  grow on their own.
* **Large gradients are chunked.** Gradients above 16.8M elements are copied, updated and
  copied back in chunks, so the three overlap. This matters most for the last gradient of
  the backward pass. In a language model that is the embedding, which the next forward pass
  needs first, so the GPU waits on exactly this tail. Chunks draw the same rounding bits as a
  whole-tensor update.
* **Small parameters are grouped.** Parameters below `small_param_threshold` (65,536 elements:
  norms, biases) get no per-tensor hook; at `step_sync()` they move as one buffer each way.
* **Exact-size pinned buffers.** PyTorch's pinned allocator rounds every buffer up to a power
  of two, so a 272 MB gradient buffer locks 512 MB. adamx registers ordinary allocations of 1 MiB
  or more with CUDA instead, which locks exactly their size.
* **Fused kernel** (`adamx/csrc/adamx_kernels.c`): one pass reads p, g, m and v and writes
  p, m, v and the low-precision copy for the GPU. ISA is chosen at runtime: AVX-512F,
  AVX2+FMA, NEON or scalar. The arithmetic is exactly `torch.optim.AdamW`'s.
* **Moment storage** (`precision=`):

  | precision | m | v | host bytes/param | kernel |
  |---|---|---|---:|---|
  | `fp32` | FP32 | FP32 | 12 | native |
  | `bf16` | BF16 | BF16 | 8 | native |
  | `fp8` | E4M3, per-128 scale | E5M2 of `sqrt(v)`, per-128 scale | 6.06 | native |

  Plus the transfer buffers: one gradient and one write-back buffer per parameter, in its dtype.
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
   on the thread split or on chunking.
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
* `small_param_threshold=65536`: CUDA parameters with fewer elements are updated together at
  `step_sync()` instead of through per-tensor hooks. `0` hooks every parameter.
* Gradient accumulation: run all but the last micro-batch under `with opt.no_sync():`.
* Global gradient clipping is impossible in async mode, because the first layers update
  before the last gradients exist. Use synchronous `step()` if you need it.
* `overlap_writeback=False` defers all weight copies to `step_sync()`. You only need it if
  something reads a parameter during backward after that parameter's gradient is complete.
* `state_dict()` saves moments in their stored precision. `load_state_dict()` also accepts
  `torch.optim.AdamW` checkpoints and converts precision when needed.
* `opt.memory_report()`, `opt.last_step_stats`, `opt.autotune_log` and `opt.backend`
  are there for inspection.
* `opt.close()` (or dropping the optimizer) stops the worker threads and releases the
  pinned buffers.

## Testing

```bash
pytest                          # 83 tests; CUDA tests run when a GPU is present
cc -O2 -pthread -o selftest tests/native/selftest.c -lm && ./selftest   # includes the kernel source
```

What has been verified:

* FP32 results equal `torch.optim.AdamW` to about 1e-7, on CPU and CUDA, in sync and async modes.
* Covered: parameter groups, schedulers, weight tying, gradient accumulation, BF16 params,
  exact checkpoint resume, chunked updates (bit-identical to whole-tensor updates),
  exact-size pinned buffers, and the auto-tuner.
* The C self-test checks every ISA path against scalar code, the FP8 codec, thread-count and
  chunking invariance, and pools used right after creation. It passes under ASan/UBSan and TSan.
* The suite runs on Windows (zig) and Linux (gcc).
* `adamx-lab` (git-ignored) runs all of this plus the benchmarks and the LLM fine-tuning
  comparison from one page, and writes the report.

## Limitations / not yet verified

* The **AVX-512 and NEON paths compile but were not executed**: the test machine is AVX2.
  Run `selftest` on such hardware before relying on them.
* **Multi-socket NUMA placement** is covered by unit tests on a simulated sysfs, not
  measured on a real two-socket server.
* The GPU waits once per step for the update of the last gradient of the backward pass (for
  language models, the embedding). Its cost grows with that tensor's size and the bytes per
  parameter of the moments, and on GPUs that drop their clocks when idle, the next passes
  run slower too.
* Not covered yet: FP16 gradients with `GradScaler`, and integration with DDP or FSDP
  sharding (single-process, one or more GPUs).
#   a d a m x  
 