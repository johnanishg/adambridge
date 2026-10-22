import sys

import pytest
import torch

from adamx import hardware


def test_parse_cpulist():
    assert hardware.parse_cpulist("0-3,8,10-11\n") == [0, 1, 2, 3, 8, 10, 11]
    assert hardware.parse_cpulist("") == []


def test_topology_from_fake_sysfs(tmp_path, monkeypatch):
    """A two-socket machine: CPUs are grouped by node and filtered by affinity."""
    for node, cpus in ((0, "0-3"), (1, "4-7")):
        d = tmp_path / "devices" / "system" / "node" / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpus)
    monkeypatch.setattr(hardware, "_SYS", tmp_path)
    monkeypatch.setattr(hardware, "_allowed_cpus", lambda: [0, 1, 2, 3, 4, 5])
    hardware.topology.cache_clear()
    try:
        topo = hardware.topology()
        assert topo.nodes == {0: (0, 1, 2, 3), 1: (4, 5)}
        assert topo.cpus(1) == (4, 5)
        assert topo.cpus(None) == (0, 1, 2, 3, 4, 5)
    finally:
        hardware.topology.cache_clear()


@pytest.mark.skipif(sys.platform == "win32", reason="PCI bus ids contain ':'")
def test_gpu_numa_node_from_fake_sysfs(tmp_path, monkeypatch):
    gpu = tmp_path / "bus" / "pci" / "devices" / "0000:41:00.0"
    gpu.mkdir(parents=True)
    (gpu / "numa_node").write_text("1\n")

    class Props:
        pci_domain_id, pci_bus_id, pci_device_id = 0, 0x41, 0

    monkeypatch.setattr(hardware, "_SYS", tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Props)
    assert hardware.gpu_numa_node(0) == 1


def test_l2_cache_has_a_sane_value():
    assert 64 * 1024 <= hardware.l2_cache_bytes() <= 64 * 1024 * 1024
