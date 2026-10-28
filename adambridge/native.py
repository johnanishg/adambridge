"""Build and load the native SIMD kernel library.

The kernels are plain C (``csrc/adambridge_kernels.c``) loaded with ctypes, so
no libtorch headers or matching compiler ABI are needed. ctypes releases the
GIL for the duration of each call, which lets the optimizer's worker threads
run the kernel truly in parallel.

The library is compiled on first use and cached under ``ADAMBRIDGE_CACHE_DIR``
(default ``~/.cache/adambridge``), keyed by a hash of the source. Compilers are
tried in this order: ``$ADAMBRIDGE_CC``, ``cc``, ``gcc``, ``clang``, the
``ziglang`` pip package (``python -m ziglang cc``), and MSVC ``cl``.

Set ``ADAMBRIDGE_DISABLE_NATIVE=1`` to force the pure-PyTorch kernels.

Command line::

    python -m adambridge build [--force] [--verbose]
    python -m adambridge info
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import logging
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = ["NativeKernels", "Group", "Segment", "load", "build", "available", "SIMD_NAMES", "BuildError"]

log = logging.getLogger("adambridge.native")

ABI_VERSION = 3
SOURCE = Path(__file__).with_name("csrc") / "adambridge_kernels.c"
SIMD_NAMES = {0: "scalar", 1: "avx2", 2: "avx512", 3: "neon"}

DT_NONE, DT_F32, DT_BF16, DT_FP8 = 0, 1, 2, 3
_DTYPE_CODES = {torch.float32: DT_F32, torch.bfloat16: DT_BF16}
MOMENT_CODES = {"fp32": DT_F32, "bf16": DT_BF16, "fp8": DT_FP8}


class BuildError(RuntimeError):
    pass


class _HParams(ctypes.Structure):
    _fields_ = [
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("one_minus_beta1", ctypes.c_float),
        ("one_minus_beta2", ctypes.c_float),
        ("decay", ctypes.c_float),
        ("step_size", ctypes.c_float),
        ("inv_sqrt_bc2", ctypes.c_float),
        ("eps", ctypes.c_float),
        ("seed", ctypes.c_uint32),
        ("stochastic", ctypes.c_int32),
    ]


class Group(ctypes.Structure):
    """Hyperparameters shared by the segments of one parameter group (adambridge_group)."""

    _fields_ = [
        ("lr", ctypes.c_double),
        ("beta1", ctypes.c_double),
        ("beta2", ctypes.c_double),
        ("eps", ctypes.c_double),
        ("weight_decay", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("stochastic", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class Segment(ctypes.Structure):
    """One tensor's host buffers (adambridge_segment). Pointers are raw addresses."""

    _fields_ = [
        ("p", ctypes.c_void_p),
        ("g", ctypes.c_void_p),
        ("m", ctypes.c_void_p),
        ("v", ctypes.c_void_p),
        ("m_scale", ctypes.c_void_p),
        ("v_scale", ctypes.c_void_p),
        ("out", ctypes.c_void_p),
        ("group", ctypes.POINTER(Group)),
        ("n", ctypes.c_int64),
        ("step", ctypes.c_int64),
        ("index", ctypes.c_uint64),
        ("offset", ctypes.c_int64),
        ("g_dtype", ctypes.c_int32),
        ("mv_dtype", ctypes.c_int32),
        ("out_dtype", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


def _lib_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def cache_dir() -> Path:
    root = os.environ.get("ADAMBRIDGE_CACHE_DIR")
    return Path(root) if root else Path.home() / ".cache" / "adambridge"


def _source_tag() -> str:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"adambridge_kernels_{plat}_{digest}"


def library_path() -> Path:
    return cache_dir() / (_source_tag() + _lib_suffix())


def _compiler_candidates() -> list[tuple[str, list[str]]]:
    """(kind, argv-prefix) pairs; kind is 'gnu' (gcc/clang-style flags) or 'msvc'."""
    found: list[tuple[str, list[str]]] = []
    env_cc = os.environ.get("ADAMBRIDGE_CC")
    if env_cc:
        argv = shlex.split(env_cc, posix=sys.platform != "win32")
        kind = "msvc" if Path(argv[0]).stem.lower() == "cl" else "gnu"
        found.append((kind, argv))
    for name in ("cc", "gcc", "clang"):
        exe = shutil.which(name)
        if exe:
            found.append(("gnu", [exe]))
    if importlib.util.find_spec("ziglang") is not None:
        found.append(("gnu", [sys.executable, "-m", "ziglang", "cc"]))
    if sys.platform == "win32" and shutil.which("cl"):
        found.append(("msvc", ["cl"]))
    return found


def _compile_cmd(kind: str, prefix: list[str], out: Path) -> list[str]:
    if kind == "msvc":
        return [*prefix, "/nologo", "/O2", "/LD", str(SOURCE), f"/Fe:{out}"]
    cmd = [*prefix, "-O3", "-std=c11", "-shared", "-fvisibility=hidden", str(SOURCE), "-o", str(out)]
    if sys.platform != "win32":
        cmd[-3:-3] = ["-fPIC", "-pthread"]
    if sys.platform.startswith("linux"):
        cmd.append("-lm")
    return cmd


def build(force: bool = False, verbose: bool = False) -> Path:
    """Compile the kernel library into the cache directory and return its path."""
    target = library_path()
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    candidates = _compiler_candidates()
    if not candidates:
        raise BuildError(
            "no C compiler found; install gcc/clang, `pip install ziglang`, "
            "or point ADAMBRIDGE_CC at a compiler"
        )
    errors = []
    for kind, prefix in candidates:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            out = Path(tmp) / target.name
            cmd = _compile_cmd(kind, prefix, out)
            if verbose:
                print("adambridge: compiling:", " ".join(cmd), file=sys.stderr)
            try:
                proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=300)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{prefix[0]}: {exc}")
                continue
            if proc.returncode != 0 or not out.exists():
                errors.append(f"{' '.join(prefix)}: exit {proc.returncode}\n{proc.stderr.strip()}")
                continue
            os.replace(out, target)  # atomic, so concurrent builders don't clash
            log.info("built %s with %s", target, " ".join(prefix))
            return target
    raise BuildError("all compilers failed:\n" + "\n\n".join(errors))


@dataclass(frozen=True)
class StepCoefficients:
    """Per-step scalars for the fused update (see ``csrc/adambridge_kernels.c``)."""

    beta1: float
    beta2: float
    decay: float
    step_size: float
    inv_sqrt_bc2: float
    eps: float

    @classmethod
    def for_step(cls, lr: float, beta1: float, beta2: float, eps: float, weight_decay: float, step: int):
        bc1 = 1.0 - beta1**step
        bc2 = 1.0 - beta2**step
        return cls(
            beta1=beta1,
            beta2=beta2,
            decay=1.0 - lr * weight_decay,
            step_size=lr / bc1,
            inv_sqrt_bc2=1.0 / bc2**0.5,
            eps=eps,
        )


class NativeKernels:
    """Thin ctypes wrapper around the compiled library."""

    def __init__(self, path: Path):
        self.path = path
        lib = ctypes.CDLL(str(path))
        lib.adambridge_abi_version.restype = ctypes.c_int
        if lib.adambridge_abi_version() != ABI_VERSION:
            raise BuildError(f"{path} has ABI {lib.adambridge_abi_version()}, expected {ABI_VERSION}")
        for fn, struct in (("adambridge_hparams_size", _HParams), ("adambridge_group_size", Group), ("adambridge_segment_size", Segment)):
            if getattr(lib, fn)() != ctypes.sizeof(struct):
                raise BuildError(f"{fn}: struct layout mismatch between C and Python")
        for fn in ("adambridge_simd_level", "adambridge_detected_simd_level", "adambridge_set_simd_level"):
            getattr(lib, fn).restype = ctypes.c_int
        lib.adambridge_set_simd_level.argtypes = [ctypes.c_int]
        lib.adambridge_fused_adamw.restype = ctypes.c_int
        lib.adambridge_fused_adamw.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(_HParams),
        ]
        lib.adambridge_pool_create.restype = ctypes.c_void_p
        lib.adambridge_pool_create.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        lib.adambridge_pool_destroy.restype = None
        lib.adambridge_pool_destroy.argtypes = [ctypes.c_void_p]
        lib.adambridge_pool_threads.restype = ctypes.c_int32
        lib.adambridge_pool_threads.argtypes = [ctypes.c_void_p]
        lib.adambridge_pool_set_spin.restype = None
        lib.adambridge_pool_set_spin.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        seg_array = ctypes.POINTER(ctypes.POINTER(Segment))
        lib.adambridge_multi_step.restype = ctypes.c_int
        lib.adambridge_multi_step.argtypes = [ctypes.c_void_p, seg_array, ctypes.c_int32, ctypes.c_int32]
        lib.adambridge_multi_apply.restype = ctypes.c_int
        lib.adambridge_multi_apply.argtypes = [ctypes.c_void_p, seg_array, ctypes.POINTER(_HParams), ctypes.c_int32, ctypes.c_int32]
        self._lib = lib

    @property
    def simd_level(self) -> str:
        return SIMD_NAMES[self._lib.adambridge_simd_level()]

    @property
    def detected_simd_level(self) -> str:
        return SIMD_NAMES[self._lib.adambridge_detected_simd_level()]

    def set_simd_level(self, name: str | None) -> str:
        """Restrict the ISA used (``None`` restores the best detected one)."""
        code = -1 if name is None else {v: k for k, v in SIMD_NAMES.items()}[name]
        return SIMD_NAMES[self._lib.adambridge_set_simd_level(code)]

    @staticmethod
    def supports(grad_dtype: torch.dtype, moment_precision: str, out_dtype: torch.dtype | None) -> bool:
        return (
            grad_dtype in _DTYPE_CODES
            and moment_precision in MOMENT_CODES
            and (out_dtype is None or out_dtype in _DTYPE_CODES)
        )

    @staticmethod
    def dtype_code(dtype: torch.dtype | None) -> int:
        return DT_NONE if dtype is None else _DTYPE_CODES[dtype]

    # -- thread pool -------------------------------------------------------

    def create_pool(self, threads: int, cpus: tuple[int, ...] | None = None) -> int:
        """Native pool of ``threads`` threads (the calling thread counts as one)."""
        arr = (ctypes.c_int32 * len(cpus))(*cpus) if cpus else None
        handle = self._lib.adambridge_pool_create(threads, arr, len(cpus) if cpus else 0)
        if not handle:
            raise RuntimeError("adambridge_pool_create failed")
        spin = os.environ.get("ADAMBRIDGE_SPIN")
        if spin:
            self._lib.adambridge_pool_set_spin(handle, int(spin))
        return handle

    def destroy_pool(self, handle: int) -> None:
        self._lib.adambridge_pool_destroy(handle)

    def pool_threads(self, handle: int) -> int:
        return self._lib.adambridge_pool_threads(handle)

    # -- batched updates ---------------------------------------------------

    @staticmethod
    def segment_array(segments: list) -> ctypes.Array:
        """Array of pointers for multi_step/multi_apply; ``segments`` holds Segment pointers."""
        return (ctypes.POINTER(Segment) * len(segments))(*segments)

    def multi_step(self, pool: int | None, segments: ctypes.Array, threads: int) -> None:
        """Update every segment; coefficients come from each segment's group and step."""
        rc = self._lib.adambridge_multi_step(pool, segments, len(segments), threads)
        if rc != 0:
            raise RuntimeError(f"adambridge_multi_step rejected a segment (code {rc})")

    def multi_apply(self, pool: int | None, segments: ctypes.Array, hparams: list, threads: int) -> None:
        """Like multi_step with explicit coefficients, one ``_HParams`` per segment."""
        hps = (_HParams * len(hparams))(*hparams)
        rc = self._lib.adambridge_multi_apply(pool, segments, hps, len(segments), threads)
        if rc != 0:
            raise RuntimeError(f"adambridge_multi_apply rejected a segment (code {rc})")

    @staticmethod
    def make_hparams(c: StepCoefficients, seed: int, stochastic: bool) -> _HParams:
        return _HParams(
            c.beta1, c.beta2, 1.0 - c.beta1, 1.0 - c.beta2,
            c.decay, c.step_size, c.inv_sqrt_bc2, c.eps,
            seed & 0xFFFFFFFF, int(stochastic),
        )

    def fused_adamw(
        self,
        master: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None,
        hp: _HParams,
        lo: int,
        hi: int,
    ) -> None:
        """Update elements ``[lo, hi)`` of flat contiguous CPU tensors in place."""
        mv_code = _DTYPE_CODES[m.dtype]
        g_code = _DTYPE_CODES[grad.dtype]
        o_code = DT_NONE if out is None else _DTYPE_CODES[out.dtype]
        rc = self._lib.adambridge_fused_adamw(
            master.data_ptr() + lo * 4,
            grad.data_ptr() + lo * grad.element_size(),
            g_code,
            m.data_ptr() + lo * m.element_size(),
            v.data_ptr() + lo * v.element_size(),
            mv_code,
            None if out is None else out.data_ptr() + lo * out.element_size(),
            o_code,
            hi - lo,
            lo,
            ctypes.byref(hp),
        )
        if rc != 0:
            raise RuntimeError(f"adambridge_fused_adamw failed with code {rc}")


_lock = threading.Lock()
_loaded: NativeKernels | None = None
_load_error: str | None = None


def load(build_if_missing: bool = True) -> NativeKernels | None:
    """Return the native kernels, compiling them if needed; ``None`` if unavailable."""
    global _loaded, _load_error
    if os.environ.get("ADAMBRIDGE_DISABLE_NATIVE", "") not in ("", "0"):
        return None
    with _lock:
        if _loaded is not None or _load_error is not None:
            return _loaded
        try:
            path = library_path()
            if not path.exists():
                if not build_if_missing:
                    return None
                path = build()
            _loaded = NativeKernels(path)
        except (BuildError, OSError) as exc:
            _load_error = str(exc)
            log.warning("adambridge native kernels unavailable, using PyTorch kernels: %s", exc)
        return _loaded


def available() -> bool:
    return load() is not None


def load_error() -> str | None:
    return _load_error


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m adambridge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile the kernel library")
    b.add_argument("--force", action="store_true")
    b.add_argument("--verbose", action="store_true")
    sub.add_parser("info", help="show the library path and SIMD level")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        print(build(force=args.force, verbose=args.verbose))
        return 0
    kernels = load()
    if kernels is None:
        print(f"native kernels unavailable: {load_error()}")
        return 1
    print(f"library:  {kernels.path}")
    print(f"simd:     {kernels.simd_level} (detected {kernels.detected_simd_level})")
    from . import hardware

    topo = hardware.topology()
    print(f"numa:     {topo.num_nodes} node(s): " + "; ".join(f"node{n}: {len(c)} cpus" for n, c in topo.nodes.items()))
    print(f"l2:       {hardware.l2_cache_bytes() // 1024} KiB")
    return 0
