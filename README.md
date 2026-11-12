# adambridge

AdamW for PyTorch with the optimizer state offloaded to host RAM and updated by
fused SIMD kernels on the CPU. It overlaps those updates with the GPU backward
pass, layer by layer. It implements the *OptiCompute-Bridge* design: the GPU
keeps only the parameters, while FP32 master weights and both Adam moments
live on the (NUMA-local) host.

```python
import adambridge

opt = adambridge.AdamBridge(model.parameters(), lr=1e-4, weight_decay=0.01,
                            precision="bf16_moments")   # 'fp32' | 'bf16' | 'fp8'
opt.attach_hooks(model)                                 # layer-wise async updates

for x, y in loader:
    loss = loss_fn(model(x), y)
    loss.backward()     # each layer is offloaded + updated as soon as its grad is ready
    opt.step_sync()     # wait; the next forward pass sees the new weights
```

Without `attach_hooks`, `AdamBridge` is a drop-in optimizer (`backward(); step(); zero_grad()`),
and works with LR schedulers and `clip_grad_norm_`. `adambridge.OptiComputeAdamW` is an alias.

Also in the box:

* **Delayed updates** (`delay_updates=True`): the host update overlaps the whole next forward and
  backward pass instead of only the backward, at the cost of one step of staleness (see
  [Delayed updates](#delayed-updates)).
* **`LionBridge`**: the Lion optimizer on the same pipeline, with one moment instead of two
  (8 host bytes per parameter with FP32 state instead of 12; see [LionBridge](#lionbridge)).
* **torch.compile**: works in every mode, including CUDA graphs and compiled autograd (see
  [torch.compile](#torchcompile)).

## Measured results

Unless a subsection names its own report (the arena and the 7B recipe do), the numbers in this section come from
two `adambridge-lab` reports generated on 2026-10-28 (`adambridge-lab/reports/20261028-224442-*`). Test machine: a laptop with an NVIDIA GeForce
RTX 5090 Laptop GPU (24 GiB VRAM), Intel Core Ultra 9 290HX Plus (24 threads, AVX2), 64 GiB RAM,
Windows 11, Python 3.14.7, PyTorch 2.14.1 + CUDA 13.2, adambridge 0.1.0 (native/avx2 backend).
Compared libraries: bitsandbytes 0.50.2, torchao 0.18.0; DeepSpeed 0.19.8 (CPU kernel only,
under WSL, because it is not installed on Windows).

### Fine-tuning a real LLM

Qwen2.5-0.5B (494M parameters) fine-tuned on WikiText-2: 200 steps of 8 × 512 tokens,
peak learning rate 2e-5 with warm-up and cosine decay. Every configuration starts from
the same weights, sees the same batches in the same order, and runs in a fresh process
(the *LLM fine-tuning* experiment in `adambridge-lab`). Validation loss is measured on the
whole WikiText-2 validation split (start: 2.868).

| configuration | final val. loss | Δ vs reference | step | peak GPU | host RAM† |
|---|---:|---:|---:|---:|---:|
| PyTorch AdamW, FP32 weights + autocast (**reference**) | 2.6402 | | 292 ms | 19.6 GiB | |
| PyTorch AdamW, BF16 weights + FP32 master, all on GPU | 2.6404 | +0.0002 | 281 ms | 18.8 GiB | <0.1 GiB |
| **adambridge, FP32 moments** | 2.6409 | +0.0008 | 395 ms | 13.2 GiB | +7.4 GiB |
| **adambridge, BF16 moments** | 2.6405 | +0.0003 | 317 ms | 13.2 GiB | +5.6 GiB |
| **adambridge, FP8 moments** | 2.6409 | +0.0007 | 305 ms | 13.2 GiB | +4.7 GiB |
| bitsandbytes AdamW8bit, FP32 weights | 2.6433 | +0.0031 | 312 ms | 16.8 GiB | <0.1 GiB |
| bitsandbytes AdamW8bit, BF16 weights | 2.6557 | +0.0156 | 298 ms | 14.2 GiB | <0.1 GiB |
| bitsandbytes PagedAdamW8bit, BF16 weights | 2.6549 | +0.0147 | 306 ms | 13.3 GiB | <0.1 GiB |
| torchao CPUOffloadOptimizer, FP32 weights | 2.6401 | −0.0001 | 525 ms | 15.9 GiB | +10.6 GiB |
| torchao CPUOffloadOptimizer, BF16 weights | 2.6526 | +0.0124 | 341 ms | 13.2 GiB | +5.3 GiB |
| PyTorch AdamW, BF16 weights only | 2.6526 | +0.0124 | 266 ms | 15.1 GiB | <0.1 GiB |

† growth of the process's resident memory beyond the all-GPU reference's.

* **Noise.** Re-running the reference with two other data orders moved its final loss by
  up to 0.0017, so differences below 0.002 are noise. All three adambridge precisions are
  inside that band, and so is the same algorithm run entirely on the GPU: offloading the
  optimizer costs no quality.
* **Why the BF16-only configurations fall behind.** An AdamW update at this learning rate is
  smaller than half the BF16 spacing of most weights, so without an FP32 copy it is rounded
  away: only 18% of their weights changed (at BF16 resolution) vs 84% for the reference.
  adambridge keeps BF16 weights on the GPU but the FP32 master on the host.
* **Memory.** adambridge needs 32% less GPU memory than the reference, the least of any
  configuration at reference quality, and 2.7 GiB less than torchao's offload at the same quality.
* **Speed.** With BF16 or FP8 moments adambridge is 5–9% slower than the all-GPU reference
  (35% with FP32 moments) and needs 40% less time per step than torchao's offload at the
  same quality (25% less with FP32 moments). The GPU waits once per step for the update of
  the last gradient of the backward pass. This laptop's GPU also drops its clocks whenever it
  idles: a forward+backward pass that takes 251 ms back to back takes 22% longer after a 20 ms
  pause and 33% longer after a 50 ms one, which every CPU-offload optimizer pays.
* **Host memory.** adambridge's host footprint is its FP32 master, the moments and one gradient and
  one write-back buffer per parameter: with FP8 moments, less than torchao's BF16 offload,
  which keeps no FP32 master.

### Against DeepSpeed, torchao, bitsandbytes and all-GPU PyTorch at 0.5B–7B (arena)

The table below is printed by `python benchmarks/readme_claims.py <report>.json` from an `adambridge-lab`
arena report, so it never says more than the lab measured. Arena report
`20261112-172201-adambridge-evaluation-complete-everything-2026-11-12-17-22.json` (generated 2026-11-12
17:22): 100 optimizer steps of 8 × 512 tokens on WikiText-2 per run, activation checkpointing on for every
library, every library's loss computed with `causal_lm_loss` (no full logits tensor for anyone), NVIDIA GeForce
RTX 5090 Laptop GPU (24 GiB) with 43 GiB of host memory (WSL 2). Each cell is the library's fastest configuration
at reference quality (the same final validation loss as the reference, within the run-to-run noise); a dash means
no configuration of it fitted or reached reference quality. DeepSpeed 0.19.9 runs in its documented single-GPU
configurations (ZeRO-2 with CPU offload, ZeRO-3 with offload of optimizer and parameters, ZenFlow, SuperOffload);
every library trains the same model with the same batches.

| model | adambridge | DeepSpeed | torchao | bitsandbytes | PyTorch | adambridge vs DeepSpeed |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B (0.5B) | 11,784 tokens/s | 5,541 tokens/s | 6,770 tokens/s | 10,640 tokens/s | 11,623 tokens/s | +113% |
| Qwen2.5-1.5B (1.5B) | 4,829 tokens/s | 2,192 tokens/s | 2,680 tokens/s | 4,341 tokens/s | – | +120% |
| Qwen2.5-3B (3.1B) | 2,782 tokens/s | – | – | – | – | only adambridge trains it |
| Qwen2.5-7B (7.6B) | 1,122 tokens/s (not judged) | – | – | – | – | no reference fits |

At 0.5B the fastest adambridge configuration (`gpu_state_budget="auto"`) ties all-GPU PyTorch (11,623 tokens/s).
At 7B only adambridge with FP8 moments and an FP24 master fits; nothing else trains it, so there is no reference
to judge it against (its validation loss falls from 2.112 to 1.967 in 100 steps).

Largest model each library trains here (random-initialised Qwen2.5-3B-shaped models with more layers, 3 steps at
micro-batch 1, bisected to ±8%):

| library | largest model | stopped by |
|---|---:|---|
| adambridge · FP8 moments · FP24 master | 8.02B | host memory |
| adambridge · FP8 moments | 6.79B | host memory |
| adambridge · FP8 moments · per-parameter transfer buffers (without the ring) | 4.17B | host memory |
| adambridge · FP32 moments | 3.39B | host memory |
| bitsandbytes AdamW8bit (FP32 weights) | 2.08B | GPU memory |
| torchao CPUOffloadOptimizer (FP32 weights) | 2.08B | host memory |
| DeepSpeed ZeRO-2 + CPU offload (ZeRO-Offload) | 2.08B | host memory |
| DeepSpeed ZeRO-3 + CPU offload of optimizer and parameters (ZeRO-Infinity) | 1.54B | host memory |
| PyTorch AdamW, all on the GPU (FP32 weights) | 1.31B | GPU memory |
| PyTorch AdamW, all on the GPU (BF16 weights + FP32 master) | 1.16B | GPU memory |

Runs whose timing the report flagged (left out above): DeepSpeed ZenFlow (ZeRO-2 + CPU offload, stall-free)
(Qwen2.5-0.5B): one step took 9 s against a median of 458 ms (the machine slept or stalled). ZenFlow also changed the model's weights before training in every run (its untrained model
evaluates to a loss of 25 instead of 2.7), so its quality is never at reference level here.

### Fine-tuning a 7B model on a 24 GB GPU

Two things stop a 7.6B-parameter model from training on a 24 GB laptop GPU with 44 GB of host memory: the full
logits tensor (8 × 512 tokens over a 152k vocabulary is 2.3 GiB in FP32, needed twice), and the host memory of
the optimizer state (6 bytes per parameter with FP8 moments and an FP32 master: 42.6 GiB). `causal_lm_loss`
removes the first and `master_precision="fp24"` cuts the second to ~5.06 bytes per parameter (36 GiB):

```python
import adambridge
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", dtype=torch.bfloat16, device_map="cuda")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
opt = adambridge.AdamBridge(model.parameters(), lr=2e-5, weight_decay=0.0, precision="fp8",
                            master_precision="fp24").attach_hooks(model)

for ids in batches:                                   # [8, 512] token ids on the GPU
    hidden = model.model(input_ids=ids).last_hidden_state
    loss = adambridge.causal_lm_loss(hidden, model.lm_head, ids, chunk_tokens=1024)
    loss.backward()
    opt.step_sync()
```

In the arena above this trains Qwen2.5-7B at 1,122 tokens/s with 18.2 GiB of GPU memory and 38.0 GiB of host
memory. Nothing fits at 7B to judge its quality against, so the FP24 master is judged where references exist, against
the same FP8 moments with an FP32 master (Δ final validation loss vs the reference; ties are within ±0.0020 nats,
±0.0031 at 3B):

| model | FP32 master | FP24 master | tokens/s (FP32 → FP24 master) | peak host memory (FP32 → FP24 master) |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | +0.0004 | +0.0002 | 11,462 → 11,601 | 4.9 → 4.4 GiB |
| Qwen2.5-1.5B | +0.0002 | +0.0002 | 4,795 → 4,829 | 10.8 → 9.4 GiB |
| Qwen2.5-3B | +0.0001 | −0.0002 | 2,691 → 2,680 | 19.5 → 16.6 GiB |

The FP24 master ties the reference at every size, at the same speed, with 9–15% less host memory.

### Fine-tuning a small LLM (quick preset)

SmolLM2-135M fine-tuned on WikiText-2: 60 steps of 8 × 256 tokens, same protocol as above
(start: 3.306). The report labels these quick-preset numbers as indicative only.

| configuration | final val. loss | Δ vs reference | step | peak GPU | host RAM† |
|---|---:|---:|---:|---:|---:|
| PyTorch AdamW, FP32 weights + autocast (**reference**) | 3.1245 | | 94 ms | 4.74 GiB | |
| PyTorch AdamW, BF16 weights + FP32 master, all on GPU | 3.1260 | +0.0015 | 89 ms | 4.38 GiB | <0.1 GiB |
| **adambridge, FP32 moments** | 3.1259 | +0.0014 | 115 ms | 2.86 GiB | +2.1 GiB |
| **adambridge, BF16 moments** | 3.1258 | +0.0013 | 106 ms | 2.86 GiB | +1.6 GiB |
| **adambridge, FP8 moments** | 3.1232 | −0.0013 | 105 ms | 2.86 GiB | +1.4 GiB |
| bitsandbytes AdamW8bit, FP32 weights | 3.1246 | +0.0001 | 109 ms | 3.98 GiB | <0.1 GiB |
| bitsandbytes AdamW8bit, BF16 weights | 3.2867 | +0.1622 | 99 ms | 3.12 GiB | <0.1 GiB |
| bitsandbytes PagedAdamW8bit, BF16 weights | 3.2867 | +0.1622 | 102 ms | 2.87 GiB | <0.1 GiB |
| torchao CPUOffloadOptimizer, FP32 weights | 3.1246 | +0.0001 | 157 ms | 3.71 GiB | +1.9 GiB |
| torchao CPUOffloadOptimizer, BF16 weights | 3.2867 | +0.1622 | 132 ms | 2.86 GiB | +1.1 GiB |
| PyTorch AdamW, BF16 weights only | 3.2866 | +0.1621 | 83 ms | 3.37 GiB | <0.1 GiB |

The reference's own spread across data orders was 0.0014, so ties are called within 0.002.
adambridge uses 40% less GPU memory than the reference at the same quality. The BF16-only
configurations kept only 10–11% of the reference's loss reduction: 2% of their weights
changed at BF16 resolution, vs 27% for the reference. This model showed no measurable GPU
pause penalty (±0% after 20 ms, +2% after 50 ms).

### Training step, speed and GPU memory

A 67M-parameter model (8 blocks, width 1024), 4,096 tokens per step, median of the timed
steps. Except the all-GPU baseline, every configuration keeps BF16 weights on the GPU.

| configuration | step | peak GPU | optimizer state |
|---|---:|---:|---|
| forward + backward only (no optimizer) | 30.2 ms | 0.891 GiB | none |
| `torch.optim.AdamW` fused, on GPU (FP32 + autocast) | 35.0 ms | 1.71 GiB | FP32 weights + FP32 m, v on GPU |
| bitsandbytes AdamW8bit | 36.5 ms | 1.02 GiB | 8-bit m, v on GPU, no FP32 master |
| bitsandbytes PagedAdamW8bit | 37.1 ms | 0.893 GiB | 8-bit m, v in unified memory, no FP32 master |
| torchao CPUOffloadOptimizer | 36.1 ms | 0.891 GiB | BF16 copy + BF16 m, v on CPU, no FP32 master |
| **adambridge async, FP32 moments** | 40.6 ms | 0.891 GiB | FP32 master + FP32 m, v on CPU |
| **adambridge async, BF16 moments** | 36.6 ms | 0.891 GiB | FP32 master + BF16 m, v on CPU |
| **adambridge async, FP8 moments** | 36.1 ms | 0.891 GiB | FP32 master + FP8 m, v on CPU |
| adambridge sync (`step()`), FP32 / BF16 / FP8 moments | 60.8 / 54.0 / 51.0 ms | 0.891 GiB | as above |

With FP8 moments adambridge is 3% slower than the all-GPU AdamW and uses 52% of its GPU memory.
The asynchronous mode (`attach_hooks`) takes 29–33% less time per step than synchronous
`step()` here.

### Host kernel

8M parameters, 24 threads (`benchmarks/bench_kernels.py --numel 8M`). At this size part of
the data stays in the CPU caches, so these times are optimistic.

| implementation | Windows ms/step | Windows GB/s | WSL ms/step | WSL GB/s |
|---|---:|---:|---:|---:|
| `torch.optim.AdamW` (foreach, CPU) | 5.83 | 38 | 7.99 | 28 |
| `torch.optim.AdamW(fused=True)` (CPU) | 2.31 | 97 | 1.98 | 113 |
| DeepSpeed CPUAdam | – | – | 2.22 | 101 |
| adambridge native, FP32 moments | 1.83 | 123 | 2.13 | 105 |
| adambridge native, FP32 moments, BF16 gradient in + BF16 weights out | 1.92 | 116 | 2.43 | 92 |
| adambridge native, BF16 moments (same I/O) | 1.13 | 141 | 1.68 | 95 |
| adambridge native, FP8 moments (same I/O) | 1.80 | 72 | 3.19 | 40 |
| adambridge PyTorch fallback, FP32 / BF16 / FP8 moments | 4.48 / 7.20 / 10.55 | | 94.57 / 13.38 / 16.77 | |

GB/s counts the minimum bytes each variant must move. The fastest adambridge variant (BF16
moments) is 2.0× faster than PyTorch's fused CPU AdamW on Windows and 1.18× under WSL.
With FP32 moments adambridge is 1.26× faster on Windows and slightly slower under WSL
(2.13 vs 1.98 ms). FP8 moments trade kernel speed for host memory; under WSL they are the slowest
native variant.

## Install

```bash
pip install .               # or: uv pip install .   /   uv add ./adambridge
pip install ".[compiler]"   # also installs ziglang, a C compiler for machines without one
python -m adambridge build       # optional: compile the native kernels now
python -m adambridge info        # shows the ISA, NUMA nodes and L2 size
```

adambridge is a pure-Python wheel (`py3-none-any`) that carries its C source, so the same
artifact installs everywhere.

**CUDA torch:** adambridge only declares `torch>=2.1`. On Windows, and for specific CUDA
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
`~/.cache/adambridge` (override with `ADAMBRIDGE_CACHE_DIR`). The build tries `$ADAMBRIDGE_CC`, then
`cc`, `gcc`, `clang`, then `python -m ziglang cc` (`pip install ziglang` gives Windows
a compiler), then MSVC `cl`. If none works, adambridge falls back to the PyTorch kernels,
which support everything but are slower. Set `ADAMBRIDGE_DISABLE_NATIVE=1` to force the fallback.

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
  of two, so a 272 MB gradient buffer locks 512 MB. adambridge registers ordinary allocations of 1 MiB
  or more with CUDA instead, which locks exactly their size.
* **Fused kernel** (`adambridge/csrc/adambridge_kernels.c`): one pass reads p, g, m and v and writes
  p, m, v and the low-precision copy for the GPU. ISA is chosen at runtime: AVX-512F,
  AVX2+FMA, NEON or scalar. The arithmetic is exactly `torch.optim.AdamW`'s.
* **Moment storage** (`precision=`):

  | precision | m | v | host bytes/param | kernel |
  |---|---|---|---:|---|
  | `fp32` | FP32 | FP32 | 12 | native |
  | `bf16` | BF16 | BF16 | 8 | native |
  | `fp8` | E4M3, per-128 scale | E5M2 of `sqrt(v)`, per-128 scale | 6.06 | native |

  With an FP32 master (4 of those bytes); `master_precision="fp24"` takes one byte off every row. Plus the
  transfer buffers: a 512 MB pinned ring per GPU for models above ~128M parameters (one gradient and one
  write-back buffer per parameter below that).
* **Master weights** (`master_precision=`): FP32, or FP24, the top 24 bits of the FP32 value kept as two
  planes (the BF16 bits, and the next byte). The kernel reads a value by joining the planes, updates it in
  FP32, rounds the result stochastically to 24 bits from its own random stream, stores the planes, and takes
  the GPU's BF16 copy from the value it stored. Every ISA does the same integer rounding, so chunked and
  threaded updates stay bit-identical to whole-tensor ones.
* **Loss without the full logits** (`adambridge.causal_lm_loss`): for a causal LM with a large vocabulary
  the logits of one batch (`batch × seq × vocab` in FP32, kept for the backward pass) can be the single largest
  tensor on the GPU: 2.3 GiB for 8 × 512 tokens of Qwen2.5. The helper takes the backbone's hidden states and
  the `lm_head`, computes the cross-entropy a slice of positions at a time under `torch.utils.checkpoint`, and
  returns the same value and gradients as the standard loss while only ever holding one slice's logits.
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
   adambridge rounds stochastically using a counter-based hash. The native and PyTorch kernels
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

## Parameters

```python
adambridge.AdamBridge(
    params,
    lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
    *,                                   # everything below is keyword-only
    precision="fp32", stochastic_rounding=True, backend="auto",
    numa_aware=True, num_threads=None, chunk_size=None,
    release_grads=True, overlap_writeback=True,
    autotune=False, seed=0, small_param_threshold=65536,
    delay_updates=False, delay_warmup_steps=0, accumulation_steps=1,
    transfer_buffer_mb=None, gpu_state_budget=None,
)
```

### AdamW hyperparameters

These behave exactly as in `torch.optim.AdamW`. They can differ between parameter groups
(pass a list of dicts as `params`) and can be changed by LR schedulers.

| parameter | default | accepted values | meaning and use |
|---|---|---|---|
| `params` | required | iterable of tensors, or of dicts (parameter groups) | The parameters to optimize. Use groups to give, say, biases and norms `weight_decay=0`. Sparse parameters are rejected. |
| `lr` | `1e-3` | float `>= 0` | Learning rate: the size of each update. Fine-tuning typically uses `1e-5`–`1e-4`. |
| `betas` | `(0.9, 0.999)` | two floats, each in `[0, 1)` | Decay rates of the running averages of the gradient (`m`) and of its square (`v`). Higher values average over more steps. |
| `eps` | `1e-8` | float `>= 0` | Added to `sqrt(v)` in the denominator so the update never divides by zero. |
| `weight_decay` | `1e-2` | float `>= 0` | Decoupled weight decay: each step every weight is shrunk by `lr * weight_decay * w`. `0` disables it. |

### Memory and precision

| parameter | default | accepted values | meaning and use |
|---|---|---|---|
| `precision` | `"fp32"` | `"fp32"` (alias `"float32"`), `"bf16"` (aliases `"bf16_moments"`, `"bfloat16"`), `"fp8"` (alias `"fp8_moments"`); case-insensitive | How the two Adam moments are stored in host RAM: 12, 8 or ~6.06 host bytes per parameter including the FP32 master (see the table in *How it works*). Lower precision uses less RAM and is faster; in the measurements above all three reached reference quality. The master weights are kept in every mode. Can be changed later with `set_precision()`. |
| `master_precision` | `"fp32"` | `"fp32"`, `"fp24"` | How the master weights are stored. `"fp24"` keeps the top 24 bits of each FP32 value (sign, 8 exponent bits, 15 mantissa bits: relative precision 2^-16) as two planes, 3 bytes per parameter instead of 4, so FP8 moments give ~5.06 host bytes per parameter. Updates are rounded stochastically (to nearest even if `stochastic_rounding=False`), and the BF16 copy sent to the GPU is always the stored value rounded, so GPU weights and master never drift apart. Fine-tuning updates (10⁻⁴ to 10⁻³ of a weight) are far above the 24-bit resolution; the lab's arena judges the quality. FP32 CPU parameters are their own master either way. Checkpoints save the planes and load into either precision. |
| `stochastic_rounding` | `True` | `True`, `False` | Rounds BF16/FP8 moments up or down at random, in proportion to the remainder, so tiny per-step changes accumulate on average. Has no effect with `precision="fp32"`. Keep it on: without it `v` stops changing once `beta2` is close to 1 (see *Where this deviates…*, item 1). |
| `seed` | `0` | int | Base seed for stochastic rounding. The same seed gives bit-identical results regardless of thread count or chunking. |
| `transfer_buffer_mb` | `None` | `None` or int `>= 0` | Pinned host memory per GPU for moving gradients to the host and new weights back. By default each large parameter gets its own pinned buffers (4 bytes per BF16 parameter, for the whole model) unless that would exceed 512 MB; then they all share a 512 MB ring instead, and backward waits briefly whenever the ring is full (which only happens when the CPU update is the bottleneck anyway). For Qwen2.5-7B (7.6B parameters) this saves about 30 GB of host RAM. An int sets the ring size in MB; `0` always uses per-parameter buffers. Delayed updates keep a 2-byte-per-parameter write-back buffer either way (it holds the update until the next step). Results are bit-identical to per-parameter buffers. |
| `gpu_state_budget` | `None` | `None`, `"auto"`, or bytes `>= 0` | Keep the optimizer state (FP32 master and moments, 12 bytes per parameter) of some parameters on the GPU, updated there by fused AdamW. The parameters chosen are the last to receive their gradients in backward (the embedding, the first blocks): their host update cannot overlap anything, so it is what each step waits for. `"auto"` uses the GPU memory left free after the first step, keeping a margin of 1 GiB or 5%; an int caps it. When the whole model fits, this is all-GPU AdamW; when nothing fits, it is the default offload. AdamW only; checkpoints keep the usual format. |

### Hardware and threading

| parameter | default | accepted values | meaning and use |
|---|---|---|---|
| `backend` | `"auto"` | `"auto"`, `"native"`, `"torch"` | Which update kernel runs on the CPU. `"auto"` uses the compiled SIMD kernel when it can be built and otherwise the PyTorch fallback. `"native"` raises an error if the kernel isn't available. `"torch"` always uses the slower PyTorch kernels, which is useful for debugging. Check the result with `opt.backend`. |
| `numa_aware` | `True` | `True`, `False` | On multi-socket Linux machines, keeps each GPU's optimizer state in the RAM attached to the CPU closest to that GPU and runs its updates on that CPU's cores. No effect on Windows, macOS, or single-socket machines. |
| `num_threads` | `None` | `None` or int `>= 1` | CPU threads per NUMA node for the update. `None` uses two thirds of the node's CPUs, capped at 32: the update is limited by memory bandwidth, so more threads stop helping well before all cores are busy, and the remaining cores keep the training loop launching GPU work (with all but two CPUs busy, forward and backward ran up to 25% slower while an update overlapped them); `tune_threads()` or `autotune` finds the best count. |
| `chunk_size` | `None` | `None` or a positive multiple of 128 | Elements processed per pass by the **PyTorch fallback** kernel. `None` derives it from the CPU's L2 cache size (between 262,144 and 4,194,304). Ignored by the native kernel. |

### Asynchronous mode (`attach_hooks`)

| parameter | default | accepted values | meaning and use |
|---|---|---|---|
| `release_grads` | `True` | `True`, `False` | Frees each gradient from GPU memory as soon as it has been copied to the host, so `p.grad` is `None` after `backward()`. Set `False` if your code reads `p.grad` after backward (for logging, say), at the cost of keeping all gradients on the GPU. |
| `overlap_writeback` | `True` | `True`, `False` | Copies each updated weight back to the GPU as soon as it is ready, overlapping with the rest of backward. `False` defers all copies to `step_sync()`. Only needed if something reads a parameter during backward after that parameter's gradient is complete (only possible with `.detach()` tricks). |
| `small_param_threshold` | `65536` | int `>= 0` | CUDA parameters with fewer elements than this (norms, biases) get no per-tensor hook and are updated together at `step_sync()`, with one transfer each way. Their update takes microseconds, while a hook costs tens of microseconds of Python. `0` hooks every parameter. |
| `autotune` | `False` | `False`, `True`, or an `AutoTuneConfig` | Tunes the thread count and moment precision while training (see below). `True` uses `AutoTuneConfig()` defaults. Decisions are made from `step_sync()` timings and recorded in `opt.autotune_log`. |
| `delay_updates` | `False` | `True`, `False` | Apply each update one step late, so the CPU update overlaps the next forward and backward pass and `step_sync()` returns at once (see [Delayed updates](#delayed-updates)). CUDA parameters only; CPU parameters are always updated in place. |
| `delay_warmup_steps` | `0` | int `>= 0` | With `delay_updates`, run this many exact steps first. |
| `accumulation_steps` | `1` | int `>= 1` | Gradient accumulation without `no_sync()`: the hooks update only on every N-th `backward()`; the gradients of the others accumulate. Needed when a training loop you don't control runs the micro-batches (the Hugging Face `Trainer`). `step()` / `step_sync()` reset the count, and pick up the gradients of a window that was cut short. |

### Delayed updates

With `delay_updates=True`, `step_sync()` queues the step's updates and returns immediately.
Each parameter's new weights are copied to the GPU during the **next** backward pass, at the
moment that parameter's next gradient is ready (backward no longer reads it then), just
before that gradient is copied out. The CPU update of step *t* therefore overlaps the whole
forward and backward pass of step *t+1*, not just the backward of step *t*. No extra memory
is used.

The price is one step of staleness: the forward pass of step *t+2* uses the weights after
step *t*'s update, as in DeepSpeed ZeRO-Offload's delayed parameter update. A parameter that
gets no gradient in a step is brought up to date at that step's `step_sync()`, so no weight
lags by more than one step.

```python
opt = adambridge.AdamBridge(model.parameters(), lr=1e-4, precision="fp32",
                            delay_updates=True, delay_warmup_steps=10).attach_hooks(model)
for x, y in loader:
    loss_fn(model(x), y).backward()
    opt.step_sync()          # returns at once
opt.flush()                  # apply pending updates before evaluating or saving the model
```

`state_dict()` and `close()` call `flush()` themselves. Training curves match AdamW applied
one step late exactly (`tests/test_cuda.py::test_delayed_updates_match_one_step_stale_adamw`).

**Staleness can cost quality, so measure it on your task.** On the lab's small synthetic language
model (400 steps), AdamW applied one step late plateaued at a held-out loss of 2.40 against 2.08
for exact AdamW. That holds for plain PyTorch as well as adambridge, at learning rates 1e-3 and
3e-3. On the lab's regression task the two were identical. Use it when the step-time gain
matters and your validation loss confirms the trade, and keep `delay_warmup_steps` for the
fast-moving first steps.

### LionBridge

`adambridge.LionBridge` runs Lion (Chen et al., 2023) on the same pipeline:

```
c = beta1 * m + (1 - beta1) * g
p = p * (1 - lr * weight_decay) - lr * sign(c)
m = beta2 * m + (1 - beta2) * g
```

```python
opt = adambridge.LionBridge(model.parameters(), lr=3e-5, betas=(0.9, 0.99), weight_decay=0.1,
                            precision="bf16").attach_hooks(model)
```

Lion keeps one moment instead of two: 8 host bytes per parameter with FP32 state (AdamW 12),
6 with BF16 and about 5 with FP8. Each update also moves fewer bytes. On the 32M-parameter
kernel benchmark below, FP32 state takes 7.0 ms (AdamW: 10.7 ms) and BF16 state 6.4 ms
(AdamW: 8.1 ms). Lion needs a learning rate typically 3–10× smaller than AdamW's, with a
correspondingly larger weight decay. Whether it converges better than AdamW depends on the
task, so tune it rather than assuming. Every `AdamBridge` keyword except `eps` applies. Every
multiply-add in the Lion kernels is fused, scalar code included, so each SIMD path should give
bit-identical results; the self-test checks this (verified for AVX2 and NEON against scalar; see the
limitations on AVX-512).

### torch.compile

The gradient hooks run outside compiled graphs and are opaque to TorchDynamo, so adambridge
works with every compile mode. The tests (`tests/test_compile.py`, Linux + CUDA + Triton)
check results against eager `torch.optim.AdamW` for:

* `torch.compile(model)`, `mode="reduce-overhead"` (CUDA graphs) and `mode="max-autotune"`;
* per-block compilation (`block.compile()` for each transformer block);
* compiled autograd (`torch._dynamo.compiled_autograd`);
* delayed updates with per-block compilation and with CUDA graphs.

**Compile per block, not the whole model.** A whole-model compile turns the backward pass
into one opaque operation, so every gradient becomes ready at its very end and the
layer-by-layer overlap is lost. Compiling each block keeps one backward operation per block,
which is also what Hugging Face recommends for compile time:

```python
for block in model.model.layers:
    block.compile()
opt = adambridge.AdamBridge(model.parameters(), precision="bf16").attach_hooks(model)
```

### Auto-tuning (`adambridge.autotune.AutoTuneConfig`)

The tuner first benchmarks thread counts once. Then, if the CPU update takes longer than the
GPU's backward pass, it steps the moment precision down one level (`fp32` → `bf16` → `fp8`).
It keeps the change only if the update actually got faster, and stops at the first decision.

```python
from adambridge.autotune import AutoTuneConfig
opt = adambridge.AdamBridge(model.parameters(), autotune=AutoTuneConfig(min_precision="fp8"))
```

| field | default | accepted values | meaning and use |
|---|---|---|---|
| `warmup_steps` | `3` | int `>= 0` | Steps ignored at start-up and after each precision change, while caches and allocations settle. |
| `window` | `5` | int `>= 1` | Steps measured (median) before each decision. |
| `threshold` | `1.0` | float `> 0` | The host counts as the bottleneck when `cpu_update > threshold * gpu_backward`. Lower it to downgrade sooner. |
| `min_precision` | `"bf16"` | any `precision` value | Lowest precision the tuner may pick. The default never selects FP8; set `"fp8"` to allow it, or `"fp32"` to disable precision tuning. |
| `min_gain` | `0.05` | float in `[0, 1)` | A downgrade is kept only if host update time drops by at least this fraction (5%); otherwise it is reverted. |
| `tune_threads` | `True` | `True`, `False` | Benchmark thread counts on the first tuned step and keep the fastest; within 3% of the best, the fewest threads win. Only applies to the native kernel. |

### Methods

| method | use |
|---|---|
| `step(closure=None)` | Synchronous update, as with any PyTorch optimizer: copy gradients to the host, update, copy weights back. Works with gradient clipping and LR schedulers. |
| `zero_grad()` | As in PyTorch. |
| `attach_hooks(model=None)` | Switches to asynchronous mode: each parameter is updated as soon as its gradient is ready during `backward()`. With `model`, only that model's parameters are hooked. Returns the optimizer. |
| `detach_hooks()` | Back to synchronous mode. |
| `step_sync()` | In asynchronous mode, call after `backward()` instead of `step()`. Updates the small parameters, waits for every queued update, and makes the next GPU work wait for the new weights. Returns `StepStats`. |
| `no_sync()` | Context manager for gradient accumulation: run all but the last micro-batch inside `with opt.no_sync():`. Without it (or `accumulation_steps`), a second gradient before `step_sync()` raises an error. |
| `flush()` | Applies every pending delayed update to the model and waits for the host. Call it before evaluating or saving with `delay_updates=True`; otherwise it only waits. |
| `set_precision(precision)` | Re-encodes all moments in another precision between steps. |
| `tune_threads(numel=4194304)` | Benchmarks the native kernel and sets the fastest thread count per device. Returns `{device: {threads: seconds}}`. |
| `state_dict()` / `load_state_dict(sd)` | Checkpointing. Moments are saved in their stored precision. Loading also accepts `torch.optim.AdamW` checkpoints and converts precision when needed. `state_dict()` applies pending delayed updates first. A `LionBridge` checkpoint has no `exp_avg_sq` and cannot be loaded into `AdamBridge`. |
| `memory_report()` | Host bytes held, by category: `master`, `moments`, `staging` (transfer buffers) and `total`. |
| `expected_bytes_per_param()` | Host bytes per parameter for master + moments at the current precisions (AdamW: 12, 8 or ~6.06 with an FP32 master, one less with FP24; Lion: 8, 6 or ~5.03). |
| `master_params()` | The FP32 master copies (`None` for a parameter until its first update; a decoded copy when the master is FP24). |
| `close()` | Applies pending delayed updates, stops the worker threads and releases pinned memory. Runs automatically (without applying) when the optimizer is garbage-collected. |

Read-only attributes: `opt.backend` (`"native/avx2"`, `"native/avx512"`, `"native/neon"`,
`"native/scalar"` or `"torch"`), `opt.precision`, `opt.autotune_log` (list of the tuner's
decisions) and `opt.last_step_stats`.

`step_sync()` and `opt.last_step_stats` return a `StepStats` with these fields (seconds):

| field | meaning |
|---|---|
| `num_updates` | Parameters updated this step (a chunked update counts once). |
| `gpu_backward` | GPU time from the first to the last gradient being ready; `None` without CUDA hooks. |
| `cpu_update` | CPU kernel time of the busiest device worker. |
| `stall` | How long `step_sync()` blocked the training loop, including waiting for the GPU to finish backward. |
| `per_device_cpu` | `cpu_update` broken down by device. |

### `adambridge.causal_lm_loss`

`causal_lm_loss(hidden_states, lm_head, labels, *, chunk_tokens=1024, ignore_index=-100)` returns the next-token
cross-entropy of a causal language model with the same value and gradients as the Hugging Face loss (labels
shifted by one, logits upcast to FP32, mean over the positions whose label is not `ignore_index`), but computes
the logits `chunk_tokens` positions at a time and recomputes each slice in the backward pass, so the memory
for them is one slice's rather than the whole batch's. `hidden_states` is `[batch, seq, hidden]` from the
backbone (`model.model(input_ids=ids).last_hidden_state` for a transformers model), `lm_head` is called as
`lm_head(x)` (a module or a function), `labels` is `[batch, seq]`. Peak memory for the logits is about
`chunk_tokens × vocab × 10` bytes. Under `torch.no_grad()` the slices are simply computed in turn. It is not
tied to adambridge's optimizer: any training loop can use it.

### Environment variables

| variable | default | meaning and use |
|---|---|---|
| `ADAMBRIDGE_CACHE_DIR` | `~/.cache/adambridge` | Where the compiled native kernel is stored. |
| `ADAMBRIDGE_CC` | unset | Compiler command to try first, e.g. `gcc`, `clang -march=native` or `cl`. |
| `ADAMBRIDGE_DISABLE_NATIVE` | unset | Any value except `0` forces the PyTorch fallback kernels. |
| `ADAMBRIDGE_SPIN` | `4000` | How long the update threads busy-wait (in CPU pause instructions) for the next job before sleeping. Higher reacts faster between updates but burns CPU; `0` sleeps immediately. |

### Command line

| command | use |
|---|---|
| `python -m adambridge build [--force] [--verbose]` | Compile the native kernel now instead of on first use. `--force` rebuilds; `--verbose` prints the compiler output. |
| `python -m adambridge info` | Shows the kernel library path, SIMD level, NUMA nodes and L2 cache size. |

### Things to know

* With `delay_updates=True`, the model's weights lag the optimizer by one step until
  `flush()`: evaluate and save after flushing.
* Global gradient clipping (`clip_grad_norm_`) is impossible in asynchronous mode, because
  the first layers update before the last gradients exist. Use synchronous `step()` if you
  need it.
* LR schedulers work in both modes: they change `lr` in the parameter groups, which is read
  at each update.
* Master weights are created on a parameter's first update, so loading pretrained weights
  after building the optimizer is safe.

## Testing

```bash
pytest                          # 195 tests; CUDA tests need a GPU, tests/test_compile.py also Triton
cc -O2 -pthread -o selftest tests/native/selftest.c -lm && ./selftest   # includes the kernel source
```

On the current code (lab report 2026-11-12 17:22): 195 passed on WSL Linux with CUDA and Triton; 186 passed
and 3 skipped on Windows with CUDA; 109 passed on WSL Linux without a GPU. The C self-test passes with gcc and
with clang (`zig cc`), under ASan + UBSan and TSan, and as an ARM (NEON) build run under QEMU user-mode
emulation (`aarch64-linux-gnu-gcc -static`, `qemu-aarch64-static`).

Results from the 2026-10-28 report:

| runtime | passed | failed | skipped | native C self-test |
|---|---:|---:|---:|---|
| Windows, CUDA (kernel built with `python -m ziglang cc`) | 82 | 0 | 1 | passed, 216 cases, avx2 |
| WSL Linux, CPU only (kernel built with `/usr/bin/cc`) | 61 | 0 | 0 | passed, 216 cases, avx2 |

What the tests check:

* adambridge's updates against `torch.optim.AdamW` on CPU and GPU (FP32 tolerance: rtol 1e-6,
  atol 1e-7), in sync and async modes.
* Parameter groups, schedulers, weight tying, gradient accumulation, gradient clipping, BF16
  params, checkpoint resume, chunked updates, exact-size pinned buffers, and the auto-tuner.
* The C self-test compares the SIMD paths against scalar code and checks the FP8 codec and
  thread-count and chunking invariance; for Lion it requires every SIMD path to be
  bit-identical to scalar code.
* Delayed updates against AdamW applied one step late (losses and weights), LionBridge against
  a plain PyTorch Lion, `accumulation_steps` against accumulated AdamW, and every
  `torch.compile` mode against eager AdamW.
* `adambridge-lab` runs all of this plus the benchmarks and the LLM fine-tuning comparison from
  one page, and writes the reports.

## Limitations / not yet verified

* The **AVX-512 path compiles but was never executed**: the test machine is AVX2 and QEMU cannot emulate
  AVX-512. Run `selftest` on such hardware before relying on it. The NEON path ran only under QEMU
  emulation, not on ARM hardware.
* **Multi-socket NUMA placement** is covered by unit tests on a simulated sysfs, not
  measured on a real two-socket server.
* Without `delay_updates`, the GPU waits once per step for the update of the last gradient of
  the backward pass (for language models, the embedding). Its cost grows with that tensor's
  size and the bytes per parameter of the moments, and on GPUs that drop their clocks when
  idle, the next passes run slower too. `delay_updates=True` removes the wait, at the cost of
  one step of staleness.
* Not covered yet: FP16 gradients with `GradScaler`, and integration with DDP or FSDP
  sharding (single-process, one or more GPUs).

## License

Copyright 2026 John Anish G. Licensed under the [Apache License, Version 2.0](LICENSE).
