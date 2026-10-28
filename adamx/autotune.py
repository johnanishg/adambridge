"""Runtime auto-tuning: moment precision and worker thread count.

Precision: after a warm-up, the tuner compares the host update time with the
GPU backward time (first to last gradient-ready event). While the host is the
bottleneck it steps the moment precision down one level (fp32 -> bf16 ->
fp8) and keeps the change only if host time actually drops; otherwise it
reverts and stops. A lower precision moves fewer bytes but does more
arithmetic per element (FP8 decodes, rescales and re-encodes every block),
so the measurement, not the byte count, decides.

Threads: a short micro-benchmark of the update kernel per compute pool,
picking the fastest thread count (memory-bound kernels usually saturate
DRAM bandwidth before all cores are busy).
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field

import torch

from .engine import StepStats, UpdateEngine
from .kernels import StepCoefficients
from .precision import PRECISIONS, canonical_precision, make_store

__all__ = ["AutoTuneConfig", "AutoTuner", "benchmark_threads"]

log = logging.getLogger("adamx")


@dataclass
class AutoTuneConfig:
    warmup_steps: int = 3       # steps ignored after start-up or a precision change
    window: int = 5             # steps per measurement (median)
    threshold: float = 1.0      # host is the bottleneck when cpu_update > threshold * gpu_backward
    min_precision: str = "bf16" # lowest precision the tuner may choose
    min_gain: float = 0.05      # keep a downgrade only if host time drops by this fraction
    tune_threads: bool = True

    def __post_init__(self):
        self.min_precision = canonical_precision(self.min_precision)


@dataclass
class AutoTuner:
    config: AutoTuneConfig = field(default_factory=AutoTuneConfig)
    done: bool = False
    threads_tuned: bool = False
    decisions: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._seen = 0
        self._samples: list[StepStats] = []
        self._baseline: tuple[str, float] | None = None  # (precision, host time) before the last downgrade

    def observe(self, stats: StepStats, precision: str) -> str | None:
        """Feed one step's timing; returns a new precision to switch to, or None."""
        if self.done or stats.num_updates == 0:
            return None
        self._seen += 1
        if self._seen <= self.config.warmup_steps:
            return None
        self._samples.append(stats)
        if len(self._samples) < self.config.window:
            return None

        cpu = statistics.median(s.cpu_update for s in self._samples)
        gpu_vals = [s.gpu_backward for s in self._samples if s.gpu_backward]
        gpu = statistics.median(gpu_vals) if gpu_vals else None
        self._samples.clear()

        if self._baseline is not None:
            prev_precision, prev_cpu = self._baseline
            self._baseline = None
            if cpu > prev_cpu * (1.0 - self.config.min_gain):
                return self._finish(
                    f"{precision} did not reduce host time ({cpu * 1e3:.1f} ms vs "
                    f"{prev_cpu * 1e3:.1f} ms); reverting to {prev_precision}",
                    prev_precision,
                )
        if gpu is None:
            return self._finish("no GPU backward timing available; keeping " + precision)
        if cpu <= self.config.threshold * gpu:
            return self._finish(f"host update {cpu * 1e3:.1f} ms hidden behind backward {gpu * 1e3:.1f} ms; keeping {precision}")

        levels = list(PRECISIONS)
        i = levels.index(precision)
        if i + 1 >= len(levels) or i + 1 > levels.index(self.config.min_precision):
            return self._finish(f"host-bound ({cpu * 1e3:.1f} ms vs {gpu * 1e3:.1f} ms) at lowest allowed precision {precision}")
        nxt = levels[i + 1]
        self._baseline = (precision, cpu)
        self._seen = 0
        self.decisions.append(f"host-bound ({cpu * 1e3:.1f} ms vs {gpu * 1e3:.1f} ms): trying {nxt} moments")
        log.info("adamx autotune: %s", self.decisions[-1])
        return nxt

    def _finish(self, reason: str, precision: str | None = None) -> str | None:
        self.done = True
        self.decisions.append(reason)
        log.info("adamx autotune: %s", reason)
        return precision


def benchmark_threads(
    engine: UpdateEngine, device: torch.device, numel: int = 1 << 22, repeats: int = 3
) -> dict[int, float]:
    """Time the update kernel for several thread counts; set the fastest. Returns {threads: seconds}."""
    pool = engine.ensure_pool(device)
    out_dtype = torch.bfloat16 if device.type == "cuda" else None
    grad_dtype = torch.bfloat16 if out_dtype else torch.float32
    if not engine.uses_native(grad_dtype, engine.precision, out_dtype):
        return {}  # the PyTorch kernel is parallelized by ATen, not by the pool

    def alloc():
        master = torch.randn(numel)
        grad = torch.randn(numel, dtype=grad_dtype) * 1e-3
        m = make_store(engine.precision, numel, "m")
        v = make_store(engine.precision, numel, "v")
        out = torch.empty(numel, dtype=out_dtype) if out_dtype else None
        return master, grad, m, v, out

    master, grad, m, v, out = pool.call(alloc)
    c = StepCoefficients.for_step(1e-4, 0.9, 0.999, 1e-8, 0.0, 10)
    candidates = sorted({k for k in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if k < pool.max_threads} | {pool.max_threads})
    timings: dict[int, float] = {}
    for k in candidates:
        pool.parallelism = k
        engine.run_kernel(master, grad, m, v, out, c, 0, pool)  # warm caches / page in
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            engine.run_kernel(master, grad, m, v, out, c, 0, pool)
            samples.append(time.perf_counter() - t0)
        timings[k] = min(samples)
    best = min(timings, key=timings.get)
    # Within 3% of the best, prefer fewer threads: leaves cores for the training loop.
    best = min(k for k, t in timings.items() if t <= timings[best] * 1.03)
    pool.parallelism = best
    log.info("adamx: %d update threads for %s (%s)", best, device,
             ", ".join(f"{k}: {t * 1e3:.2f} ms" for k, t in timings.items()))
    return timings
