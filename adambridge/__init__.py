"""adambridge: AdamW (and Lion) with host-offloaded state, fused SIMD CPU
kernels, and layer-wise overlap of optimizer updates with the GPU backward pass."""

from . import hardware, native, precision
from .autotune import AutoTuneConfig
from .engine import StepStats
from .losses import causal_lm_loss
from .optimizer import AdamBridge, LionBridge, OptiComputeAdamW
from .precision import bytes_per_param

__version__ = "0.2.0"

__all__ = [
    "AdamBridge",
    "LionBridge",
    "OptiComputeAdamW",
    "AutoTuneConfig",
    "StepStats",
    "bytes_per_param",
    "causal_lm_loss",
    "hardware",
    "native",
    "precision",
    "__version__",
]
