"""Storage formats for the AdamW moment vectors.

The master weights always stay FP32. The first and second moments can be
stored compressed to reduce host memory traffic, which is what bounds a
host-side optimizer step:

==========  =================  ====================  ===============
precision   first moment m     second moment v       bytes / param*
==========  =================  ====================  ===============
fp32        FP32               FP32                  12
bf16        BF16               BF16                  8
fp8         E4M3 + block scale E5M2 + block scale    ~6.06
==========  =================  ====================  ===============

\\* master weight + m + v, excluding gradient and write-back buffers.

Low-precision moments are written with *stochastic rounding* by default.
With round-to-nearest, an EMA with beta2 = 0.999 stalls: the per-step change
``(1 - beta2) * (g^2 - v)`` is smaller than half a BF16 ulp (2^-9 relative)
whenever ``g^2`` is within ~2x of ``v``, so ``v`` stops tracking the
gradient. Stochastic rounding keeps every update unbiased in expectation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = [
    "PRECISIONS",
    "canonical_precision",
    "MomentStore",
    "FP32Store",
    "BF16Store",
    "FP8Store",
    "make_store",
    "store_from_state_dict",
    "bytes_per_param",
    "round_to_bf16_stochastic",
    "round_to_fp8_stochastic",
    "moment_seed",
    "random_bits",
]

# Accepted spellings -> canonical name. Ordered from highest to lowest precision.
_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "bf16": "bf16",
    "bf16_moments": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "fp8_moments": "fp8",
}
PRECISIONS = ("fp32", "bf16", "fp8")

FP8_BLOCK = 128
_FP8_FORMATS = {
    # role -> (dtype, max finite value, mantissa bits)
    "m": (torch.float8_e4m3fn, 448.0, 3),
    "v": (torch.float8_e5m2, 57344.0, 2),
}


def canonical_precision(name: str) -> str:
    try:
        return _ALIASES[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown precision {name!r}; expected one of {sorted(_ALIASES)}"
        ) from None


def bytes_per_param(precision: str) -> float:
    """Host bytes per parameter for master weight + both moments."""
    precision = canonical_precision(precision)
    moment = {"fp32": 4.0, "bf16": 2.0, "fp8": 1.0 + 4.0 / FP8_BLOCK}[precision]
    return 4.0 + 2 * moment


# --------------------------------------------------------------------------
# Stochastic rounding primitives
#
# Random bits come from a counter-based hash of (seed, element index): the
# same lowbias32 hash the native kernel uses. It is stateless, so results do
# not depend on how work is split into chunks or threads, the PyTorch and
# native kernels make identical rounding decisions, and it is ~10x cheaper
# than torch.randint on CPU.
# --------------------------------------------------------------------------

def _hash32(x: int) -> int:
    x &= 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x


def moment_seed(seed: int, role: str) -> int:
    """32-bit rounding stream for moment ``role`` (matches the native kernel)."""
    seed &= 0xFFFFFFFF
    return seed if role == "m" else _hash32(seed ^ 0xA511E9B3)


def _to_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= 1 << 31 else value


def random_bits(offset: int, n: int, seed: int) -> torch.Tensor:
    """lowbias32(index ^ seed) for indices ``offset .. offset+n-1``, as raw int32 bit patterns."""
    if offset + n <= 1 << 31:
        x = torch.arange(offset, offset + n, dtype=torch.int32)
    else:
        x = (torch.arange(offset, offset + n, dtype=torch.int64) & 0xFFFFFFFF).to(torch.int32)
    x.bitwise_xor_(_to_i32(seed))
    # int32 shifts are arithmetic; masking makes them logical. Products wrap mod 2^32.
    x.bitwise_xor_(x.bitwise_right_shift(16).bitwise_and_(0xFFFF))
    x.mul_(0x7FEB352D)
    x.bitwise_xor_(x.bitwise_right_shift(15).bitwise_and_(0x1FFFF))
    x.mul_(_to_i32(0x846CA68B))
    x.bitwise_xor_(x.bitwise_right_shift(16).bitwise_and_(0xFFFF))
    return x


def _stochastic_truncate_(x: torch.Tensor, dropped_bits: int, offset: int, seed: int) -> torch.Tensor:
    """Stochastically round FP32 ``x`` in place to ``23 - dropped_bits`` mantissa bits.

    Adding uniform noise below the kept bits and truncating rounds up with
    probability equal to the discarded fraction. Works on sign-magnitude
    floats because integer addition on the raw bits moves the magnitude.
    """
    noise = random_bits(offset, x.numel(), seed).bitwise_and_((1 << dropped_bits) - 1)
    x.view(torch.int32).add_(noise).bitwise_and_(-(1 << dropped_bits))
    return x


def round_to_bf16_stochastic(x: torch.Tensor, seed: int, offset: int = 0) -> torch.Tensor:
    """Return FP32 ``x`` stochastically rounded to BF16. ``x`` is clobbered.

    ``offset`` is the global index of ``x[0]`` in the random stream.
    """
    return _stochastic_truncate_(x, 16, offset, seed).to(torch.bfloat16)


def round_to_fp8_stochastic(x: torch.Tensor, dtype: torch.dtype, seed: int, offset: int = 0) -> torch.Tensor:
    """Return pre-scaled FP32 ``x`` stochastically rounded to ``dtype``. ``x`` is clobbered.

    Values in the FP8 subnormal range fall back to round-to-nearest.
    """
    _, fmax, mbits = next(f for f in _FP8_FORMATS.values() if f[0] == dtype)
    _stochastic_truncate_(x, 23 - mbits, offset, seed)
    return x.clamp_(-fmax, fmax).to(dtype)


# --------------------------------------------------------------------------
# Moment stores
# --------------------------------------------------------------------------

class MomentStore(ABC):
    """One moment vector for one parameter, stored flat on the host.

    ``read(lo, hi)`` returns an FP32 tensor for elements ``[lo, hi)``; the
    caller may modify it in place and must hand it back via ``write``. For
    FP32 storage ``read`` returns a view, so updates land in place and
    ``write`` is free.
    """

    precision: str

    def __init__(self, numel: int):
        self.numel = numel

    @abstractmethod
    def read(self, lo: int, hi: int) -> torch.Tensor: ...

    @abstractmethod
    def write(self, lo: int, hi: int, x: torch.Tensor, seed: int | None = None) -> None:
        """Store FP32 ``x`` (clobbered) into ``[lo, hi)``: stochastically rounded
        with stream ``seed``, or round-to-nearest-even if ``seed`` is None."""

    @abstractmethod
    def to_fp32(self) -> torch.Tensor: ...

    @property
    @abstractmethod
    def nbytes(self) -> int: ...

    @property
    def raw(self) -> torch.Tensor:
        """The storage tensor handed to the native kernel (FP32/BF16 only)."""
        raise NotImplementedError

    @property
    def native_buffers(self) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """(codes or values, per-block scales or None) in the native kernel's
        layout, or None if this store can only be updated by the PyTorch kernel."""
        return None

    @property
    def chunk_alignment(self) -> int:
        """``lo`` passed to ``read``/``write`` must be a multiple of this."""
        return 1

    @abstractmethod
    def state_dict(self) -> dict[str, Any]: ...


class FP32Store(MomentStore):
    precision = "fp32"

    def __init__(self, numel: int, init: torch.Tensor | None = None):
        super().__init__(numel)
        self.data = torch.zeros(numel, dtype=torch.float32)
        if init is not None:
            self.data.copy_(init.reshape(-1))

    def read(self, lo, hi):
        return self.data[lo:hi]

    def write(self, lo, hi, x, seed=None):
        dst = self.data[lo:hi]
        if x.data_ptr() != dst.data_ptr():
            dst.copy_(x)

    def to_fp32(self):
        return self.data.clone()

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def raw(self):
        return self.data

    @property
    def native_buffers(self):
        return self.data, None

    def state_dict(self):
        return {"precision": "fp32", "data": self.data}


class BF16Store(MomentStore):
    precision = "bf16"

    def __init__(self, numel: int, init: torch.Tensor | None = None):
        super().__init__(numel)
        self.data = torch.zeros(numel, dtype=torch.bfloat16)
        if init is not None:
            self.data.copy_(init.reshape(-1))

    def read(self, lo, hi):
        return self.data[lo:hi].float()

    def write(self, lo, hi, x, seed=None):
        if seed is not None:
            self.data[lo:hi].copy_(round_to_bf16_stochastic(x, seed, lo))
        else:
            self.data[lo:hi].copy_(x)  # torch casts with round-to-nearest-even

    def to_fp32(self):
        return self.data.float()

    @property
    def nbytes(self):
        return self.data.nbytes

    @property
    def raw(self):
        return self.data

    @property
    def native_buffers(self):
        return self.data, None

    def state_dict(self):
        return {"precision": "bf16", "data": self.data}


class FP8Store(MomentStore):
    """FP8 with one FP32 scale per ``block`` elements (absmax scaling).

    Per-block scaling keeps each block's largest magnitude at the top of the
    FP8 range, which is what prevents small moments from underflowing to zero.

    The second moment is stored as ``sqrt(v)``. ``v`` scales with ``g^2``, so
    storing it directly would squander half of E5M2's exponent range: an
    element whose gradient is 2^16 smaller than its block's largest would
    flush ``v`` to 0 while ``m`` stayed non-zero, turning the update into
    ``m / eps``. In the square-root domain ``m`` underflows first. The cost
    is a small upward bias in ``v`` (stochastic rounding is unbiased for
    ``sqrt(v)``, so ``E[v] = v + Var``): about 1-2% with beta2 = 0.999,
    which slightly shortens steps rather than destabilizing them.
    """

    precision = "fp8"

    def __init__(self, numel: int, role: str, init: torch.Tensor | None = None, block: int = FP8_BLOCK):
        super().__init__(numel)
        if role not in _FP8_FORMATS:
            raise ValueError(f"role must be 'm' or 'v', got {role!r}")
        self.role = role
        self.block = block
        self.dtype, self.fmax, _ = _FP8_FORMATS[role]
        self.sqrt_domain = role == "v"
        nblocks = max(1, math.ceil(numel / block))
        self.q = torch.zeros(nblocks * block, dtype=self.dtype)
        self.scale = torch.ones(nblocks, dtype=torch.float32)
        if init is not None:
            self.write(0, numel, init.reshape(-1).float().clone())

    @property
    def chunk_alignment(self):
        return self.block

    def read(self, lo, hi):
        b0, b1 = lo // self.block, math.ceil(hi / self.block)
        x = self.q[b0 * self.block : b1 * self.block].float().view(-1, self.block)
        x.mul_(self.scale[b0:b1, None])
        if self.sqrt_domain:
            x.square_()
        return x.view(-1)[: hi - lo]

    def write(self, lo, hi, x, seed=None):
        n = hi - lo
        b0, b1 = lo // self.block, math.ceil(hi / self.block)
        if self.sqrt_domain:
            x = x.clamp_(min=0).sqrt_()
        padded = (b1 - b0) * self.block
        if padded != n:
            x = torch.nn.functional.pad(x, (0, padded - n))
        xb = x.view(-1, self.block)
        amax = xb.abs().amax(dim=1)
        scale = torch.where(amax > 0, amax / self.fmax, torch.ones_like(amax))
        xb.div_(scale[:, None])
        flat = xb.view(-1)
        if seed is not None:
            q = round_to_fp8_stochastic(flat, self.dtype, seed, b0 * self.block)
        else:
            q = flat.clamp_(-self.fmax, self.fmax).to(self.dtype)
        self.q[b0 * self.block : b1 * self.block].copy_(q)
        self.scale[b0:b1].copy_(scale)

    def to_fp32(self):
        return self.read(0, self.numel).clone()

    @property
    def nbytes(self):
        return self.q.nbytes + self.scale.nbytes

    @property
    def native_buffers(self):
        # The native kernel implements the 128-element block layout only.
        return (self.q, self.scale) if self.block == FP8_BLOCK else None

    def state_dict(self):
        return {"precision": "fp8", "role": self.role, "block": self.block, "q": self.q, "scale": self.scale}


def make_store(precision: str, numel: int, role: str, init: torch.Tensor | None = None) -> MomentStore:
    """Allocate a moment store. ``role`` is ``'m'`` or ``'v'`` (picks the FP8 format)."""
    precision = canonical_precision(precision)
    if precision == "fp32":
        return FP32Store(numel, init)
    if precision == "bf16":
        return BF16Store(numel, init)
    return FP8Store(numel, role, init)


def store_from_state_dict(state: dict[str, Any], numel: int, role: str) -> MomentStore:
    precision = state["precision"]
    if precision == "fp8":
        store = FP8Store(numel, state.get("role", role), block=state["block"])
        store.q.copy_(state["q"])
        store.scale.copy_(state["scale"])
        return store
    store = make_store(precision, numel, role)
    store.data.copy_(state["data"])
    return store
