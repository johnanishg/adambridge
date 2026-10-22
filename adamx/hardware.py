"""Host topology discovery: NUMA nodes, GPU locality, thread pinning, caches.

Everything here degrades gracefully: on platforms without the Linux sysfs
interfaces the machine is treated as a single NUMA node and pinning is a
no-op.
"""

from __future__ import annotations

import functools
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

__all__ = [
    "Topology",
    "topology",
    "parse_cpulist",
    "gpu_numa_node",
    "pin_current_thread",
    "l2_cache_bytes",
    "simd_capability",
]

_SYS = Path("/sys")


def parse_cpulist(text: str) -> list[int]:
    """Parse a kernel cpulist such as ``"0-3,8,10-11"``."""
    cpus: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return cpus


def _allowed_cpus() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


@dataclass(frozen=True)
class Topology:
    nodes: dict[int, tuple[int, ...]]  # NUMA node -> CPUs this process may use

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def cpus(self, node: int | None) -> tuple[int, ...]:
        if node is None or node not in self.nodes:
            return tuple(c for cpus in self.nodes.values() for c in cpus)
        return self.nodes[node]


@functools.lru_cache(maxsize=1)
def topology() -> Topology:
    allowed = set(_allowed_cpus())
    node_root = _SYS / "devices" / "system" / "node"
    nodes: dict[int, tuple[int, ...]] = {}
    if node_root.is_dir():
        for entry in sorted(node_root.glob("node[0-9]*")):
            try:
                cpus = parse_cpulist((entry / "cpulist").read_text())
            except OSError:
                continue
            usable = tuple(c for c in cpus if c in allowed)
            if usable:
                nodes[int(entry.name[4:])] = usable
    if not nodes:
        nodes = {0: tuple(sorted(allowed))}
    return Topology(nodes)


def gpu_numa_node(device: torch.device | int) -> int | None:
    """NUMA node a CUDA device's PCIe root port is attached to, if known."""
    if not torch.cuda.is_available():
        return None
    index = device if isinstance(device, int) else (device.index or 0)
    try:
        props = torch.cuda.get_device_properties(index)
        bus_id = "{:04x}:{:02x}:{:02x}.0".format(
            props.pci_domain_id, props.pci_bus_id, props.pci_device_id
        )
    except (AttributeError, RuntimeError):
        return None
    path = _SYS / "bus" / "pci" / "devices" / bus_id / "numa_node"
    try:
        node = int(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def pin_current_thread(cpus: tuple[int, ...] | list[int]) -> bool:
    """Bind the calling thread to ``cpus``. Returns False where unsupported.

    On Linux ``sched_setaffinity(0, ...)`` applies to the calling thread only,
    and memory the thread first touches afterwards is allocated on its node.
    """
    if not cpus or not hasattr(os, "sched_setaffinity"):
        return False
    try:
        os.sched_setaffinity(0, cpus)
        return True
    except OSError:
        return False


@functools.lru_cache(maxsize=1)
def l2_cache_bytes() -> int:
    """Per-core L2 size, falling back to 1 MiB when it cannot be read."""
    cache = _SYS / "devices" / "system" / "cpu" / "cpu0" / "cache"
    if cache.is_dir():
        for index in sorted(cache.glob("index*")):
            try:
                if (index / "level").read_text().strip() != "2":
                    continue
                size = (index / "size").read_text().strip().upper()
            except OSError:
                continue
            mult = {"K": 1 << 10, "M": 1 << 20}.get(size[-1], 1)
            return int(size.rstrip("KM")) * mult
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.run(["sysctl", "-n", "hw.perflevel0.l2cachesize"], capture_output=True, text=True)
            return int(out.stdout.strip())
        except (OSError, ValueError):
            pass
    return 1 << 20


def simd_capability() -> str:
    """Best SIMD level the native kernels will use, or torch's view of the CPU."""
    from . import native

    kernels = native.load(build_if_missing=False)
    if kernels is not None:
        return kernels.simd_level
    return str(torch.backends.cpu.get_cpu_capability()).lower()
