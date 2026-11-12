"""The transfer ring: pinned staging shared by every large CUDA parameter instead of buffers per parameter."""

import copy

import pytest
import torch

import adambridge
from adambridge import engine, native
from test_cuda import TinyLM, lm_loss

pytestmark = pytest.mark.cuda


def _train(model, opt, steps, hooks=True):
    losses = []
    for k in range(steps):
        loss = lm_loss(model, k)
        loss.backward()
        if hooks:
            opt.step_sync()
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    opt.flush()
    torch.cuda.synchronize()
    return losses


def _run(base, budget, steps=6, hooks=True, **kw):
    model = copy.deepcopy(base)
    opt = adambridge.AdamBridge(model.parameters(), lr=3e-3, small_param_threshold=2000, transfer_buffer_mb=0, **kw)
    opt._engine.transfer_buffer_bytes = budget  # bytes (the public option is in MB); 0: one buffer per parameter
    if hooks:
        opt.attach_hooks(model)
    losses = _train(model, opt, steps, hooks)
    return model, opt, losses


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp8"])
@pytest.mark.parametrize("delay", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ring_matches_per_parameter_buffers(monkeypatch, precision, delay, dtype):
    """A ring far smaller than one parameter (every hook waits for a lease, the ring wraps around and chunks
    of one parameter share it) gives bit-identical weights, master copies and losses."""
    kernels = native.load()
    if kernels is None or not kernels.supports(dtype, precision, dtype):
        pytest.skip("the ring needs the native kernel for this combination")
    monkeypatch.setattr(engine, "_CHUNK", 4096)  # most hooked parameters move in several chunks
    leased = []
    acquire = engine._TransferRing.acquire
    monkeypatch.setattr(engine._TransferRing, "acquire", lambda self, n: leased.append(n) or acquire(self, n))
    torch.manual_seed(0)
    base = TinyLM().cuda().to(dtype)
    want_model, want_opt, want_losses = _run(base, 0, precision=precision, delay_updates=delay)
    assert not leased                           # budget 0: no ring
    model, opt, losses = _run(base, 1, precision=precision, delay_updates=delay)  # 1 byte: the smallest ring
    ring = next(w.ring for w in opt._engine._workers.values())
    assert ring is not None and ring.capacity < max(p.numel() for p in model.parameters()) * dtype.itemsize * 2
    # Far more went through the ring than it holds: its memory was reused many times (whether a hook also had to
    # wait for a lease depends on timing, so it is not asserted).
    assert sum(leased) > 4 * ring.capacity
    assert not ring.leases or all(x.released for x in ring.leases)
    hooked = [s for s in opt._ordered if s.small is None]
    assert hooked and (delay or all(s.out is None and s.grad_host is None for s in hooked))
    assert losses == want_losses
    for p, q in zip(model.parameters(), want_model.parameters()):
        assert torch.equal(p, q)
    for s, t in zip(opt._ordered, want_opt._ordered):
        assert torch.equal(s.master.to_fp32(), t.master.to_fp32())
    opt.close()
    want_opt.close()


def test_ring_in_step_mode_and_memory_report(monkeypatch):
    """step() without hooks also uses the ring; memory_report counts the ring instead of per-parameter buffers."""
    if native.load() is None:
        pytest.skip("needs the native kernel")
    monkeypatch.setattr(engine, "_CHUNK", 4096)
    torch.manual_seed(0)
    base = TinyLM(dim=128).cuda().to(torch.bfloat16)
    want_model, want_opt, want_losses = _run(base, 0, hooks=False)
    model, opt, losses = _run(base, 1, hooks=False)
    assert losses == want_losses
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), want_model.parameters()))
    with_ring, without = opt.memory_report(), want_opt.memory_report()
    assert opt._engine.transfer_bytes() > 0 and want_opt._engine.transfer_bytes() == 0
    assert with_ring["staging"] < without["staging"]
    assert with_ring["master"] == without["master"] and with_ring["moments"] == without["moments"]
    opt.close()
    want_opt.close()


def test_ring_off_when_per_parameter_buffers_are_smaller():
    """The default ring (512 MB) is not created for a model whose buffers would take less."""
    torch.manual_seed(0)
    model = TinyLM().cuda()
    opt = adambridge.AdamBridge(model.parameters(), lr=1e-3, small_param_threshold=0).attach_hooks(model)
    _train(model, opt, 2)
    assert opt._engine.transfer_bytes() == 0
    assert all(s.out is not None for s in opt._ordered)
    opt.close()


def test_ring_unit():
    """FIFO leases, wrap-around, and waiting for the oldest lease's copy."""
    ring = engine._TransferRing(engine._PinnedBuffers(), 4 * engine._RING_ALIGN)
    a = ring.acquire(engine._RING_ALIGN)
    b = ring.acquire(2 * engine._RING_ALIGN)
    assert (a.offset, b.offset) == (0, engine._RING_ALIGN)
    ring.release(a)
    c = ring.acquire(engine._RING_ALIGN)      # after b, at the end
    assert c.offset == 3 * engine._RING_ALIGN
    d = ring.acquire(engine._RING_ALIGN)      # wraps into a's space
    assert d.offset == 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)          # the "copy" still running when b is released
        ev = torch.cuda.Event()
        ev.record(stream)
    ring.release(b, ev)
    e = ring.acquire(2 * engine._RING_ALIGN)  # must wait for ev, then take b's space
    assert ev.query() and e.offset == engine._RING_ALIGN and ring.waits == 1
    with pytest.raises(ValueError):
        ring.acquire(5 * engine._RING_ALIGN)
