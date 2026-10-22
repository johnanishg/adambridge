"""Build and load the native SIMD kernel library.

The kernels are plain C (``csrc/adamx_kernels.c``) loaded with ctypes, so
no libtorch headers or matching compiler ABI are needed. ctypes releases the
GIL for the duration of each call, which lets the optimizer's worker threads
run the kernel truly in parallel.

The library is compiled on first use and cached under ``ADAMX_CACHE_DIR``
(default ``~/.cache/adamx``), keyed by a hash of the source. Compilers are
tried in this order: ``$ADAMX_CC``, ``cc``, ``gcc``, ``clang``, the
``ziglang`` pip package (``python -m ziglang cc``), and MSVC ``cl``.

Set ``ADAMX_DISABLE_NATIVE=1`` to force the pure-PyTorch kernels.

Command line::

    python -m adamx build [--force] [--verbose]
    python -m adamx info
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

__all__ = ["NativeKernels", "load", "build", "available", "SIMD_NAMES", "BuildError"]

log = logging.getLogger("adamx.native")

ABI_VERSION = 1
SOURCE = Path(__file__).with_name("csrc") / "adamx_kernels.c"
SIMD_NAMES = {0: "scalar", 1: "avx2", 2: "avx512", 3: "neon"}

DT_NONE, DT_F32, DT_BF16 = 0, 1, 2
_DTYPE_CODES = {torch.float32: DT_F32, torch.bfloat16: DT_BF16}


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


def _lib_suffix() -> str:
    if sys.platform == "win32":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def cache_dir() -> Path:
    root = os.environ.get("ADAMX_CACHE_DIR")
    return Path(root) if root else Path.home() / ".cache" / "adamx"


def _source_tag() -> str:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"adamx_kernels_{plat}_{digest}"


def library_path() -> Path:
    return cache_dir() / (_source_tag() + _lib_suffix())


def _compiler_candidates() -> list[tuple[str, list[str]]]:
    """(kind, argv-prefix) pairs; kind is 'gnu' (gcc/clang-style flags) or 'msvc'."""
    found: list[tuple[str, list[str]]] = []
    env_cc = os.environ.get("ADAMX_CC")
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
        cmd.insert(-3, "-fPIC")
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
            "or point ADAMX_CC at a compiler"
        )
    errors = []
    for kind, prefix in candidates:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            out = Path(tmp) / target.name
            cmd = _compile_cmd(kind, prefix, out)
            if verbose:
                print("adamx: compiling:", " ".join(cmd), file=sys.stderr)
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
    """Per-step scalars for the fused update (see ``csrc/adamx_kernels.c``)."""

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
        lib.adamx_abi_version.restype = ctypes.c_int
        if lib.adamx_abi_version() != ABI_VERSION:
            raise BuildError(f"{path} has ABI {lib.adamx_abi_version()}, expected {ABI_VERSION}")
        if lib.adamx_hparams_size() != ctypes.sizeof(_HParams):
            raise BuildError("adamx_hparams layout mismatch between C and Python")
        for fn in ("adamx_simd_level", "adamx_detected_simd_level", "adamx_set_simd_level"):
            getattr(lib, fn).restype = ctypes.c_int
        lib.adamx_set_simd_level.argtypes = [ctypes.c_int]
        lib.adamx_fused_adamw.restype = ctypes.c_int
        lib.adamx_fused_adamw.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int64, ctypes.c_uint64, ctypes.POINTER(_HParams),
        ]
        self._lib = lib

    @property
    def simd_level(self) -> str:
        return SIMD_NAMES[self._lib.adamx_simd_level()]

    @property
    def detected_simd_level(self) -> str:
        return SIMD_NAMES[self._lib.adamx_detected_simd_level()]

    def set_simd_level(self, name: str | None) -> str:
        """Restrict the ISA used (``None`` restores the best detected one)."""
        code = -1 if name is None else {v: k for k, v in SIMD_NAMES.items()}[name]
        return SIMD_NAMES[self._lib.adamx_set_simd_level(code)]

    @staticmethod
    def supports(grad_dtype: torch.dtype, moment_precision: str, out_dtype: torch.dtype | None) -> bool:
        return (
            grad_dtype in _DTYPE_CODES
            and moment_precision in ("fp32", "bf16")
            and (out_dtype is None or out_dtype in _DTYPE_CODES)
        )

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
        rc = self._lib.adamx_fused_adamw(
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
            raise RuntimeError(f"adamx_fused_adamw failed with code {rc}")


_lock = threading.Lock()
_loaded: NativeKernels | None = None
_load_error: str | None = None


def load(build_if_missing: bool = True) -> NativeKernels | None:
    """Return the native kernels, compiling them if needed; ``None`` if unavailable."""
    global _loaded, _load_error
    if os.environ.get("ADAMX_DISABLE_NATIVE", "") not in ("", "0"):
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
            log.warning("adamx native kernels unavailable, using PyTorch kernels: %s", exc)
        return _loaded


def available() -> bool:
    return load() is not None


def load_error() -> str | None:
    return _load_error


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m adamx")
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
