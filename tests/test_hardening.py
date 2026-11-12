"""Hardening: checkpoint validation, bounds checks in front of the C kernel, and safe native-library loading."""

import os
import sys

import pytest
import torch

import adambridge
from adambridge import native
from adambridge.precision import make_store, store_from_state_dict


def _trained(precision="fp32"):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Linear(32, 4))
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-2, precision=precision, small_param_threshold=0)
    for _ in range(2):
        model(torch.randn(8, 16)).pow(2).mean().backward()
        opt.step()
        opt.zero_grad()
    return model, opt


def _fresh(model, precision="fp32"):
    return adambridge.AdamBridge(model.parameters(), lr=1e-2, precision=precision, small_param_threshold=0)


def test_checkpoint_with_wrong_sizes_is_rejected_before_anything_changes():
    model, opt = _trained()
    sd = opt.state_dict()
    bad = {**sd, "state": {k: dict(v) for k, v in sd["state"].items()}}
    bad["state"][0]["master"] = torch.zeros(1)  # copy_ would have broadcast it over the whole parameter
    other = _fresh(model)
    with pytest.raises(ValueError, match="elements"):
        other.load_state_dict(bad)
    other.load_state_dict(sd)  # still loadable: nothing was half-applied
    opt.close()
    other.close()


@pytest.mark.parametrize("index", [-1, 99, "x"])
def test_checkpoint_with_unknown_parameter_index_is_rejected(index):
    model, opt = _trained()
    sd = opt.state_dict()
    sd = {**sd, "state": {index: sd["state"][0]}}
    other = _fresh(model)
    with pytest.raises(ValueError, match="parameter"):
        other.load_state_dict(sd)
    opt.close()
    other.close()


def test_checkpoint_string_indices_still_load():
    model, opt = _trained()
    sd = opt.state_dict()
    sd = {**sd, "state": {str(k): v for k, v in sd["state"].items()}}
    other = _fresh(model)
    other.load_state_dict(sd)
    opt.close()
    other.close()


def test_moment_store_state_is_validated():
    n = 300
    with pytest.raises(ValueError, match="precision"):
        store_from_state_dict({"precision": "int4", "data": torch.zeros(n)}, n, "m")
    with pytest.raises(ValueError, match="elements"):
        store_from_state_dict({"precision": "fp32", "data": torch.zeros(n - 1)}, n, "m")
    fp8 = make_store("fp8", n, "m").state_dict()
    for block in (0, -128, 2.5, True):
        with pytest.raises(ValueError, match="block"):
            store_from_state_dict({**fp8, "block": block}, n, "m")
    with pytest.raises(ValueError, match="role"):
        store_from_state_dict({**fp8, "role": "w"}, n, "m")
    with pytest.raises(ValueError, match="elements"):
        store_from_state_dict({**fp8, "scale": torch.ones(1)}, n, "m")


def test_fp8_store_saved_under_the_other_role_is_converted():
    n = 256
    x = torch.rand(n) * 1e-3
    m_store = make_store("fp8", n, "m", init=x)
    loaded = store_from_state_dict(m_store.state_dict(), n, "v")  # this moment is a v store: E5M2, sqrt domain
    assert loaded.role == "v"
    # Decoded with the format it was saved in, then re-encoded as a v store (not reinterpreted as one).
    want = make_store("fp8", n, "v", init=m_store.to_fp32()).to_fp32()
    assert torch.equal(loaded.to_fp32(), want)


def test_fused_adamw_checks_bounds_and_layout():
    kernels = native.load()
    if kernels is None:
        pytest.skip("native kernels unavailable")
    n = 64
    master, grad, m, v = torch.zeros(n), torch.zeros(n), torch.zeros(n), torch.zeros(n)
    hp = kernels.make_hparams(native.StepCoefficients.for_step(1e-3, 0.9, 0.999, 1e-8, 0.0, 1), 0, False)
    kernels.fused_adamw(master, grad, m, v, None, hp, 0, n)
    with pytest.raises(IndexError):
        kernels.fused_adamw(master, grad, m, v, None, hp, 0, n + 1)
    with pytest.raises(ValueError, match="elements"):
        kernels.fused_adamw(master, torch.zeros(n // 2), m, v, None, hp, 0, n // 2)
    with pytest.raises(ValueError, match="contiguous"):
        kernels.fused_adamw(master, torch.zeros(1).expand(n), m, v, None, hp, 0, n)


def test_compilers_are_not_taken_from_the_current_directory(tmp_path, monkeypatch):
    name = "gcc.exe" if sys.platform == "win32" else "gcc"
    planted = tmp_path / name
    planted.write_text("")
    planted.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", os.environ.get("PATH", "")]))
    found = native._which("gcc")
    assert found is None or os.path.dirname(os.path.abspath(found)) != str(tmp_path)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and permission bits")
def test_library_writable_by_others_is_not_loaded(tmp_path):
    lib = tmp_path / "lib.so"
    lib.write_bytes(b"")
    tmp_path.chmod(0o755)
    lib.chmod(0o755)
    assert native._untrusted(lib) is None
    lib.chmod(0o777)
    assert "writable" in native._untrusted(lib)
    lib.chmod(0o755)
    tmp_path.chmod(0o777)
    assert "writable" in native._untrusted(lib)
    tmp_path.chmod(0o755)
    with pytest.raises(native.BuildError, match="refusing"):
        lib.chmod(0o666)
        native.NativeKernels(lib)
