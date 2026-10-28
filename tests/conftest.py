import pytest
import torch

from adambridge import native

CUDA = torch.cuda.is_available()
NATIVE = native.load() is not None

BACKENDS = ["torch"] + (["native"] if NATIVE else [])


def pytest_collection_modifyitems(config, items):
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "cuda" in item.keywords and not CUDA:
            item.add_marker(skip_cuda)


def tiny_mlp(seed=0, din=48, hidden=193, dout=5, dtype=torch.float32, device="cpu"):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(din, hidden),
        torch.nn.GELU(),
        torch.nn.LayerNorm(hidden),
        torch.nn.Linear(hidden, dout),
    ).to(device=device, dtype=dtype)


def max_param_diff(a, b):
    return max((x.float() - y.float()).abs().max().item() for x, y in zip(a.parameters(), b.parameters()))
