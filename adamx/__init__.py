"""adamx: AdamW with host-offloaded state, fused SIMD CPU kernels, and
layer-wise overlap of optimizer updates with the GPU backward pass."""

from . import hardware, native, precision
from .autotune import AutoTuneConfig
from .engine import StepStats
from .optimizer import AdamX, OptiComputeAdamW
from .precision import bytes_per_param

__version__ = "0.1.0"

__all__ = [
    "AdamX",
    "OptiComputeAdamW",
    "AutoTuneConfig",
    "StepStats",
    "bytes_per_param",
    "hardware",
    "native",
    "precision",
    "__version__",
]
