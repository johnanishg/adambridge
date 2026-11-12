"""Storage formats for the master weights and the AdamW moment vectors.

The first and second moments can be stored compressed to reduce host memory
traffic, which is what bounds a host-side optimizer step:

==========  =================  ====================  ===============
precision   first moment m     second moment v       bytes / param*
==========  =================  ====================  ===============
fp32        FP32               FP32                  12
bf16        BF16               BF16                  8
fp8         E4M3 + block scale E5M2 + block scale    ~6.06
==========  =================  ====================  ===============

\\* with an FP32 master weight; excluding gradient and write-back buffers.

The master weights are FP32 (``master_precision="fp32"``) or FP24: the top 24
bits of the FP32 value (sign, 8 exponent bits, 15 mantissa bits, relative
precision 2^-16) kept as two planes, the BF16 bits and the next byte, 3 bytes
per parameter instead of 4. An FP24 master takes one byte per parameter off
every row of the table above.

Low-precision moments (and the FP24 master) are written with *stochastic
rounding* by default. With round-to-nearest, an EMA with beta2 = 0.999
stalls: the per-step change ``(1 - beta2) * (g^2 - v)`` is smaller than half
a BF16 ulp (2^-9 relative) whenever ``g^2`` is within ~2x of ``v``, so ``v``
stops tracking the gradient. Stochastic rounding keeps every update unbiased
in expectation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import torch

__all__ = [
    "PRECISIONS",
    "MASTER_PRECISIONS",
    "canonical_precision",
    "canonical_master_precision",
    "MomentStore",
    "FP32Store",
    "BF16Store",
    "FP8Store",
    "FP24Store",
    "make_store",
    "make_master",
    "store_from_state_dict",
    "check_master_state",
    "load_master_",
    "master_to_fp32",
    "bytes_per_param",
    "round_to_bf16_stochastic",
    "round_to_fp8_stochastic",
    "round_to_fp24_stochastic",
    "round_to_fp24_nearest_",
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
_MASTER_ALIASES = {"fp32": "fp32", "float32": "fp32", "fp24": "fp24", "float24": "fp24"}
MASTER_PRECISIONS = ("fp32", "fp24")

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


def canonical_master_precision(name: str) -> str:
    try:
        return _MASTER_ALIASES[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"unknown master precision {name!r}; expected one of {sorted(_MASTER_ALIASES)}"
        ) from None


def bytes_per_param(precision: str, moments: int = 2, master: str = "fp32") -> float:
    """Host bytes per parameter for the master weight + ``moments`` moment vectors
    (AdamW: 2; Lion: 1, its single moment stored like AdamW's first)."""
    precision = canonical_precision(precision)
    moment = {"fp32": 4.0, "bf16": 2.0, "fp8": 1.0 + 4.0 / FP8_BLOCK}[precision]
    return {"fp32": 4.0, "fp24": 3.0}[canonical_master_precision(master)] + moments * moment


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
    """32-bit rounding stream for ``role``: ``'m'``, ``'v'`` or ``'p'`` (the FP24 master); matches the native
    kernel."""
    seed &= 0xFFFFFFFF
    if role == "m":
        return seed
    return _hash32(seed ^ (0xA511E9B3 if role == "v" else 0x6C8E9CF5))


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


def round_to_fp24_stochastic(x: torch.Tensor, seed: int, offset: int = 0) -> torch.Tensor:
    """Stochastically round FP32 ``x`` in place to its top 24 bits (the FP24 master format)."""
    return _stochastic_truncate_(x, 8, offset, seed)


def round_to_fp24_nearest_(x: torch.Tensor) -> torch.Tensor:
    """Round FP32 ``x`` in place to its top 24 bits, to nearest even (NaNs stay NaN), as the native kernel does."""
    b = x.view(torch.int32)
    nan = x.isnan()
    quiet = b[nan].bitwise_or_(0x00400000) if nan.any() else None
    b.add_(b.bitwise_right_shift(8).bitwise_and_(1).add_(0x7F)).bitwise_and_(~0xFF)
    if quiet is not None:
        b[nan] = quiet
    return x


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

    @classmethod
    def wrap(cls, data: torch.Tensor) -> "FP32Store":
        """A store over an existing flat FP32 tensor (a CPU parameter updated in place)."""
        store = cls.__new__(cls)
        MomentStore.__init__(store, data.numel())
        store.data = data
        return store

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


class FP24Store(MomentStore):
    """The FP24 master: the top 24 bits of each FP32 value as two planes, ``hi``
    (the BF16 bits, kept as a bfloat16 tensor so ``hi.float()`` is the exact
    expansion) and ``lo`` (the next byte). Reads expand to FP32; writes round
    stochastically with the ``'p'`` stream, or to nearest even without a seed.
    """

    precision = "fp24"
    INIT_CHUNK = 1 << 24  # elements converted at a time when initialising from FP32 (bounds the temporaries)

    def __init__(self, numel: int, init: torch.Tensor | None = None):
        super().__init__(numel)
        self.hi = torch.zeros(numel, dtype=torch.bfloat16)
        self.lo = torch.zeros(numel, dtype=torch.uint8)
        if init is not None:
            self.init_from_(init)

    def init_from_(self, src: torch.Tensor) -> None:
        """Fill the planes from ``src`` (any device or shape). A BF16 source is exact and needs no temporaries:
        its bits are the hi plane and the lo plane is zero. Anything else is rounded to nearest even, a chunk at
        a time, so a 7B model's embedding does not need a second FP32 copy of itself on the host."""
        src = src.detach().reshape(-1)
        if src.dtype == torch.bfloat16:
            self.hi.copy_(src)
            self.lo.zero_()
            return
        for lo in range(0, self.numel, self.INIT_CHUNK):
            hi = min(lo + self.INIT_CHUNK, self.numel)
            self.write(lo, hi, src[lo:hi].to("cpu", torch.float32, copy=True))

    # Little-endian layout of an FP32 element: byte 1 holds bits 8..15 (the lo plane), bytes 2..3 the BF16 bits
    # (the hi plane). Strided views move the planes without temporaries.
    def read(self, lo, hi):
        x = self.hi[lo:hi].float()  # exact: the BF16 bits on top, zeros below
        x.view(torch.uint8)[1::4].copy_(self.lo[lo:hi])
        return x

    def write(self, lo, hi, x, seed=None):
        """Round contiguous FP32 ``x`` in place to 24 bits and store it: afterwards ``x`` equals the stored
        value."""
        if seed is not None:
            round_to_fp24_stochastic(x, seed, lo)
        else:
            round_to_fp24_nearest_(x)
        self.hi[lo:hi].view(torch.int16).copy_(x.view(torch.int16)[1::2])
        self.lo[lo:hi].copy_(x.view(torch.uint8)[1::4])

    def to_fp32(self):
        return self.read(0, self.numel)

    @property
    def nbytes(self):
        return self.hi.nbytes + self.lo.nbytes

    @property
    def native_buffers(self):
        return self.hi, self.lo

    def state_dict(self):
        return {"precision": "fp24", "hi": self.hi, "lo": self.lo}


def make_store(precision: str, numel: int, role: str, init: torch.Tensor | None = None) -> MomentStore:
    """Allocate a moment store. ``role`` is ``'m'`` or ``'v'`` (picks the FP8 format)."""
    precision = canonical_precision(precision)
    if precision == "fp32":
        return FP32Store(numel, init)
    if precision == "bf16":
        return BF16Store(numel, init)
    return FP8Store(numel, role, init)


def make_master(precision: str, numel: int, init: torch.Tensor | None = None) -> MomentStore:
    """Allocate a master-weight store (``'fp32'`` or ``'fp24'``), initialised from ``init`` (any device/dtype)."""
    precision = canonical_master_precision(precision)
    if precision == "fp24":
        return FP24Store(numel, init)
    return FP32Store(numel, init)


def check_master_state(saved: Any, numel: int, what: str) -> None:
    """Validate a checkpoint's master weights: an FP32 tensor, or an FP24 store's state dict."""
    if isinstance(saved, torch.Tensor):
        _checked(saved, numel, what)
        return
    if not isinstance(saved, dict) or saved.get("precision") not in MASTER_PRECISIONS:
        raise ValueError(f"checkpoint {what}: expected a tensor or an fp32/fp24 store, got "
                         f"{saved.get('precision') if isinstance(saved, dict) else type(saved).__name__!r}")
    if saved["precision"] == "fp24":
        if _checked(saved.get("hi"), numel, f"{what} (hi plane)").dtype != torch.bfloat16:
            raise ValueError(f"checkpoint {what}: the hi plane must be bfloat16")
        if _checked(saved.get("lo"), numel, f"{what} (lo plane)").dtype != torch.uint8:
            raise ValueError(f"checkpoint {what}: the lo plane must be uint8")
    else:
        _checked(saved.get("data"), numel, what)


def master_to_fp32(saved: Any, numel: int) -> torch.Tensor:
    """The checkpoint master weights (see ``check_master_state``) as a flat FP32 CPU tensor."""
    if isinstance(saved, torch.Tensor):
        return saved.detach().reshape(-1).to("cpu", torch.float32)
    if saved["precision"] == "fp24":
        store = FP24Store(numel)
        store.hi.copy_(saved["hi"].reshape(-1))
        store.lo.copy_(saved["lo"].reshape(-1))
        return store.to_fp32()
    return saved["data"].detach().reshape(-1).to("cpu", torch.float32)


def load_master_(store: MomentStore, saved: Any) -> None:
    """Load validated checkpoint master weights into ``store`` (converting between fp32 and fp24 if needed)."""
    if isinstance(store, FP24Store) and isinstance(saved, dict) and saved.get("precision") == "fp24":
        store.hi.copy_(saved["hi"].reshape(-1))
        store.lo.copy_(saved["lo"].reshape(-1))
        return
    x = master_to_fp32(saved, store.numel)
    if isinstance(store, FP32Store):
        store.data.copy_(x)
        return
    src = saved if isinstance(saved, torch.Tensor) else saved.get("data")
    if isinstance(src, torch.Tensor) and x.data_ptr() == src.data_ptr():
        x = x.clone()  # write rounds in place: never the checkpoint's own tensor
    store.write(0, store.numel, x)


def _checked(t: Any, numel: int, what: str) -> torch.Tensor:
    """``t`` if it is a tensor of exactly ``numel`` elements (``copy_`` would silently broadcast a smaller one)."""
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"checkpoint {what} is {type(t).__name__}, expected a tensor")
    if t.numel() != numel:
        raise ValueError(f"checkpoint {what} has {t.numel()} elements, expected {numel} (a different model?)")
    return t


def store_from_state_dict(state: dict[str, Any], numel: int, role: str) -> MomentStore:
    """Rebuild a moment store from a checkpoint, validating every field (a checkpoint may come from another model,
    another version or an untrusted source)."""
    if not isinstance(state, dict) or state.get("precision") not in PRECISIONS:
        raise ValueError(f"checkpoint moment store has unknown precision {state.get('precision') if isinstance(state, dict) else state!r}")
    precision = state["precision"]
    if precision == "fp8":
        saved_role, block = state.get("role", role), state.get("block")
        if saved_role not in _FP8_FORMATS:
            raise ValueError(f"checkpoint FP8 store has unknown role {saved_role!r}")
        if not isinstance(block, int) or isinstance(block, bool) or not 0 < block <= 1 << 20:
            raise ValueError(f"checkpoint FP8 store has invalid block size {block!r}")
        store = FP8Store(numel, saved_role, block=block)
        store.q.copy_(_checked(state.get("q"), store.q.numel(), "FP8 codes"))
        store.scale.copy_(_checked(state.get("scale"), store.scale.numel(), "FP8 scales"))
        if saved_role != role:  # decode with the format it was written in, store in the one this moment uses
            store = FP8Store(numel, role, init=store.to_fp32())
        return store
    store = make_store(precision, numel, role)
    store.data.copy_(_checked(state.get("data"), numel, f"{precision} moments"))
    return store
