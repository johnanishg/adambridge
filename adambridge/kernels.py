"""Fused AdamW update kernels operating on flat host buffers.

Two implementations share one contract: update ``[lo, hi)`` of the FP32
master weights and both moment stores in place, optionally writing the new
weights (cast to the accelerator dtype) into ``out``.

* :mod:`adambridge.native` - single-pass SIMD kernel (FP32/BF16 moments).
* :func:`fused_adamw_chunk` - PyTorch ops over one chunk at a time. Each op
  is a separate pass, but over a chunk that fits in the cores' combined L2
  (ATen splits every op over its intra-op threads the same way, so each
  core revisits the same slice), keeping DRAM traffic close to one read and
  one write per element. Handles every precision, including FP8 moments.
"""

from __future__ import annotations

import math

import torch

from .native import StepCoefficients
from .precision import MomentStore, moment_seed

__all__ = ["StepCoefficients", "fused_adamw_chunk", "split_range", "chunk_ranges", "mix_seed"]

_MASK64 = (1 << 64) - 1


def mix_seed(*values: int) -> int:
    """Deterministically combine integers into a 63-bit seed (splitmix64 rounds)."""
    h = 0x9E3779B97F4A7C15
    for v in values:
        h = ((h ^ (v & _MASK64)) * 0xBF58476D1CE4E5B9) & _MASK64
        h ^= h >> 31
        h = (h * 0x94D049BB133111EB) & _MASK64
        h ^= h >> 29
    return h & ((1 << 63) - 1)


def split_range(n: int, parts: int, align: int) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into at most ``parts`` contiguous ranges whose starts are multiples of ``align``."""
    if n <= 0:
        return []
    parts = max(1, min(parts, math.ceil(n / align)))
    per = math.ceil(n / parts / align) * align
    return [(lo, min(lo + per, n)) for lo in range(0, n, per)]


def chunk_ranges(lo: int, hi: int, chunk: int) -> list[tuple[int, int]]:
    return [(a, min(a + chunk, hi)) for a in range(lo, hi, chunk)]


def fused_adamw_chunk(
    master: torch.Tensor,
    grad: torch.Tensor,
    m_store: MomentStore,
    v_store: MomentStore,
    out: torch.Tensor | None,
    c: StepCoefficients,
    lo: int,
    hi: int,
    seed: int | None = None,
) -> None:
    """AdamW on elements ``[lo, hi)`` with the same arithmetic as ``torch.optim.AdamW``.

    ``seed`` enables stochastic rounding of low-precision moments.
    """
    p = master[lo:hi]
    g = grad[lo:hi]
    if g.dtype != torch.float32:
        g = g.float()
    m = m_store.read(lo, hi)
    v = v_store.read(lo, hi)

    p.mul_(c.decay)
    m.mul_(c.beta1).add_(g, alpha=1.0 - c.beta1)
    v.mul_(c.beta2).addcmul_(g, g, value=1.0 - c.beta2)
    denom = v.sqrt().mul_(c.inv_sqrt_bc2).add_(c.eps)
    p.addcdiv_(m, denom, value=-c.step_size)

    m_store.write(lo, hi, m, None if seed is None else moment_seed(seed, "m"))
    v_store.write(lo, hi, v, None if seed is None else moment_seed(seed, "v"))
    if out is not None:
        out[lo:hi].copy_(p)
