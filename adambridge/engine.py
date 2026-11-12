"""Host update engine: the layer-wise asynchronous offload pipeline.

For a parameter on a CUDA device, one update flows through::

    autograd thread          D2H stream            worker thread            H2D stream
    ---------------          ----------            -------------            ----------
    grad ready (hook)
    record `ready` ------->  wait `ready`
                             grad -> pinned host
                             record `copied` ----> wait `copied` (CPU)
                                                   fused AdamW on host
                                                   (native thread pool)
                                                   ------------------------> wait `ready`
                                                                             pinned -> param
    ...backward continues on the compute stream for earlier layers...

``finish_step`` waits for every queued update and makes the caller's current
stream wait on the H2D streams, so the next forward pass sees new weights.

There is one worker thread per device. Gradient copies complete in order on
the D2H stream, so the worker updates, as one batch, every queued gradient
whose copy has already landed: one native call (the native library splits
the work over its own NUMA-pinned thread pool and computes each tensor's
coefficients from its step counter) and one ``_foreach_copy_`` back. Hooks
fire when autograd's CPU side reaches a parameter, which can be long before
the GPU produces its gradient; taking only completed copies keeps the host
update right behind the GPU's backward pass instead of waiting for the end of
a batch. Under load the batches grow on their own, amortizing per-call cost.
Per-tensor cost on the Python side is a few pointer writes, not a thread
hand-off.

Gradients larger than ``_CHUNK`` elements move and update in chunks: each
chunk's update starts when its copy lands, and its write-back overlaps the
next chunk's update. The last gradient of the backward pass is usually the
largest (the embedding, which the next forward pass needs first), so this
shortens the time the GPU waits between steps. Chunk updates draw the same
rounding bits as a whole-tensor update (the native segment's element offset).

Parameters below a size threshold (norms, biases) skip per-tensor hooks
altogether: a :class:`SmallGroup` gathers their gradients with one
``torch.cat`` at ``step_sync``, moves them with one copy each way and
scatters the new weights with one fused GPU copy. Their update takes
microseconds, so they lose nothing by not overlapping with backward, and a
model's many tiny tensors cost O(1) GPU calls per step instead of O(n).

**Delayed updates** (``delay = True``): ``finish_step`` does not wait for the
host. The update computed from step t's gradient stays in the slot's ``out``
buffer and is written to the parameter during step t+1's backward pass, at
the moment that parameter's next gradient is ready (backward no longer reads
it then), just before the gradient is copied out::

    hook (step t+1)        H2D stream               D2H stream      worker
    ---------------        ----------               ----------      ------
    wait until step t's
    update of p is done
    record `ready` ----->  wait `ready`
                           out_t -> p
                           record `written` ---------------------->  wait `written`
                     ----------------------------> wait `ready`
                                                    grad -> pinned
                                                    record `copied`->  wait `copied`
                                                                       update t+1 -> out

The host update of step t then overlaps the whole forward and backward pass
of step t+1 instead of only the backward of step t, at the cost of one step
of staleness: the forward pass of step t+2 uses weights that include
gradients up to step t. No extra memory is needed: one ``out`` buffer per
parameter suffices because the hook orders the write-back before the next
update. Parameters on the CPU are updated in place and are never delayed.

The engine also runs Lion (``algorithm='lion'``): one moment per parameter,
no ``v``; everything else (staging, chunking, threading) is shared.
"""

from __future__ import annotations

import collections
import ctypes
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from . import hardware, native
from .kernels import StepCoefficients, chunk_ranges, fused_adamw_chunk, fused_lion_chunk, mix_seed
from .precision import FP8_BLOCK, FP32Store, MomentStore, make_master, make_store

__all__ = ["UpdateEngine", "ParamSlot", "SmallGroup", "StepStats"]

log = logging.getLogger("adambridge")

# Ranges handed to threads start on this boundary: a multiple of the SIMD
# width (so vector/tail assignment is independent of the split) and of the
# FP8 block size.
_ALIGN = 128
_MASK64 = (1 << 64) - 1
# Gradients above this many elements are copied and updated in chunks of it
# (a multiple of FP8_BLOCK; ~1 ms per chunk over PCIe, a few ms to update).
_CHUNK = 1 << 24
# Transfer buffers of at least this many bytes are pinned at their exact size.
_REGISTER_MIN = 1 << 20


class _PinnedBuffers:
    """Page-locked host buffers of exactly the requested size.

    PyTorch's pinned allocator rounds every allocation up to a power of two:
    a 272 MB gradient buffer locks 512 MB of RAM. Large buffers are instead
    allocated as ordinary memory and registered with CUDA, which locks exactly
    their pages. Registered buffers are kept alive here until ``release()``,
    so their memory is never freed while still registered."""

    def __init__(self):
        self._registered: list[torch.Tensor] = []
        self._lock = threading.Lock()

    def empty(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        nbytes = numel * dtype.itemsize
        if nbytes >= _REGISTER_MIN:
            t = torch.empty(numel, dtype=dtype)
            rt = torch.cuda.cudart()
            if rt.cudaHostRegister(t.data_ptr(), nbytes, 0) == rt.cudaError.success:
                with self._lock:
                    self._registered.append(t)
                return t
        return torch.empty(numel, dtype=dtype, pin_memory=True)

    def free(self, t: torch.Tensor | None) -> None:
        """Give one buffer back early (no copy may still be using it)."""
        if t is None:
            return
        with self._lock:
            keep = [b for b in self._registered if b.data_ptr() != t.data_ptr()]
            found = len(keep) != len(self._registered)
            self._registered = keep
        if found:
            torch.cuda.cudart().cudaHostUnregister(t.data_ptr())

    def release(self) -> None:
        """Unregister every buffer once no copy can still be using it."""
        with self._lock:
            buffers, self._registered = self._registered, []
        if not buffers:
            return
        try:
            for device in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device)
            rt = torch.cuda.cudart()
            for t in buffers:
                rt.cudaHostUnregister(t.data_ptr())
        except Exception:  # interpreter shutdown: the process is going away anyway
            pass


_RING_ALIGN = 512
# Default size of a device's transfer ring (pinned memory shared by every large CUDA parameter's gradient copy and
# write-back). A device whose large parameters need less than this keeps one buffer per parameter instead.
RING_DEFAULT_BYTES = 512 << 20


class _Lease:
    __slots__ = ("offset", "nbytes", "released", "event")

    def __init__(self, offset: int, nbytes: int):
        self.offset, self.nbytes = offset, nbytes
        self.released = False
        self.event: torch.cuda.Event | None = None


class _TransferRing:
    """Pinned staging memory for one device, handed out first-in first-out.

    Instead of a pinned gradient buffer and a pinned write-back buffer per
    parameter (4 bytes per BF16 parameter, for the whole model), every
    gradient chunk takes a lease on the ring for its copy to the host and its
    updated weights, and returns it once the weights are back on the GPU.
    Leases are taken in backward order and returned in the same order (one
    worker per device, one copy stream each way), so a ring is enough. When
    the ring is full, the hook that wants a lease waits for the oldest one:
    backward is held back while the CPU catches up, which costs nothing when
    the CPU update is the bottleneck anyway, and never happens when it is not."""

    def __init__(self, pinned: _PinnedBuffers, capacity: int):
        self.capacity = capacity
        self.arena = pinned.empty(capacity, torch.uint8)
        self.leases: collections.deque[_Lease] = collections.deque()
        self.cv = threading.Condition(threading.Lock())
        self.waits = 0  # times a lease had to wait (backpressure), for diagnostics

    @staticmethod
    def round_up(nbytes: int) -> int:
        return -(-nbytes // _RING_ALIGN) * _RING_ALIGN

    def _reclaim(self) -> None:
        q = self.leases
        while q and q[0].released and (q[0].event is None or q[0].event.query()):
            q.popleft()

    def _fit(self, n: int) -> int | None:
        q = self.leases
        if not q:
            return 0 if n <= self.capacity else None
        tail, last = q[0].offset, q[-1]
        end = last.offset + last.nbytes
        if last.offset >= tail:  # not wrapped: free space after the newest lease, then before the oldest
            if self.capacity - end >= n:
                return end
            return 0 if tail >= n else None
        return end if tail - end >= n else None  # wrapped: free space between the newest and the oldest

    def acquire(self, nbytes: int) -> _Lease:
        n = self.round_up(nbytes)
        if n > self.capacity:
            raise ValueError(f"a {nbytes}-byte transfer does not fit the {self.capacity}-byte ring")
        with self.cv:
            waited = False
            while True:
                self._reclaim()
                off = self._fit(n)
                if off is not None:
                    lease = _Lease(off, n)
                    self.leases.append(lease)
                    self.waits += waited
                    return lease
                waited = True
                oldest = self.leases[0]
                if oldest.released and oldest.event is not None:
                    event = oldest.event
                    self.cv.release()
                    try:
                        event.synchronize()
                    finally:
                        self.cv.acquire()
                else:
                    self.cv.wait(0.05)

    def view(self, lease: _Lease, dtype: torch.dtype, numel: int, byte_offset: int = 0) -> torch.Tensor:
        if byte_offset < 0 or byte_offset + numel * dtype.itemsize > lease.nbytes:
            raise ValueError(f"view of {numel} x {dtype} at +{byte_offset} B exceeds its {lease.nbytes} B lease")
        lo = lease.offset + byte_offset
        return self.arena[lo:lo + numel * dtype.itemsize].view(dtype)

    def release(self, lease: _Lease, event: torch.cuda.Event | None = None) -> None:
        """The lease's memory is free once ``event`` (if any) has completed."""
        with self.cv:
            if lease.released:
                return
            lease.event = event
            lease.released = True
            self.cv.notify_all()


class ParamSlot:
    """Host-side optimizer state for one parameter."""

    __slots__ = (
        "index", "param", "group", "numel", "step",
        "master", "out", "out_view", "m", "v",
        "grad_host", "submitted",
        "seg", "seg_ptr", "seg_group", "ready_ev", "copied_evs", "written_ev", "small",
        "inflight", "gpu",
    )

    def __init__(self, index: int, param: torch.Tensor, group: dict):
        self.index = index
        self.param = param
        self.group = group
        self.numel = param.numel()
        self.step = 0
        self.master: MomentStore | None = None    # flat, host: FP32Store (maybe the CPU parameter itself) or FP24Store
        self.out: torch.Tensor | None = None      # write-back buffer in param dtype (pinned for CUDA)
        self.out_view: torch.Tensor | None = None  # ``out`` shaped like the parameter
        self.m: MomentStore | None = None
        self.v: MomentStore | None = None         # None for Lion
        self.grad_host: torch.Tensor | None = None  # pinned gradient staging (CUDA params)
        self.submitted = False
        self.seg: native.Segment | None = None    # native description of the buffers above
        self.seg_ptr = None
        self.seg_group = None
        self.ready_ev: torch.cuda.Event | None = None
        self.copied_evs: list[torch.cuda.Event] = []  # one per chunk of the gradient copy
        self.written_ev: torch.cuda.Event | None = None  # delayed mode: `out` copied to the parameter
        self.small: SmallGroup | None = None      # set if updated as part of a group
        self.inflight = 0                         # queued update tasks not yet run (engine._cv)
        self.gpu: torch.Tensor | None = None      # FP32 master on the GPU: state kept there (gpu_state_budget)

    @property
    def initialized(self) -> bool:
        return self.master is not None


class SmallGroup:
    """Small CUDA parameters of one device and dtype, moved as one buffer.

    ``out_flat`` (pinned) holds every member's write-back buffer back to back;
    members' ``out`` are views into it, so the new weights go to the GPU in
    one copy and are scattered with one fused ``_foreach_copy_``.
    """

    def __init__(self, slots: list[ParamSlot]):
        self.slots = slots
        self.device = slots[0].param.device
        self.dtype = slots[0].param.dtype
        self.offsets = []
        total = 0
        for slot in slots:
            self.offsets.append(total)
            total += slot.numel
            slot.small = self
        self.total = total
        self.out_flat = torch.empty(total, dtype=self.dtype, pin_memory=True)
        self._grad_host: dict[torch.dtype, torch.Tensor] = {}
        self._gpu_flat: torch.Tensor | None = None
        self._gpu_views: list[torch.Tensor] = []

    def out_for(self, slot: ParamSlot) -> torch.Tensor:
        i = self.slots.index(slot)
        return self.out_flat[self.offsets[i]:self.offsets[i] + slot.numel]

    def grad_host(self, dtype: torch.dtype) -> torch.Tensor:
        buf = self._grad_host.get(dtype)
        if buf is None:
            buf = self._grad_host[dtype] = torch.empty(self.total, dtype=dtype, pin_memory=True)
        return buf

    def write_back(self) -> None:
        """Copy every member's new weights to its parameter (on the current stream)."""
        if self._gpu_flat is None:
            self._gpu_flat = torch.empty(self.total, dtype=self.dtype, device=self.device)
            self._gpu_views = [self._gpu_flat[o:o + s.numel].view(s.param.shape)
                               for o, s in zip(self.offsets, self.slots)]
        self._gpu_flat.copy_(self.out_flat, non_blocking=True)
        torch._foreach_copy_([s.param.data for s in self.slots], self._gpu_views)


@dataclass
class StepStats:
    """Timing for one optimizer step (seconds)."""

    num_updates: int = 0
    gpu_backward: float | None = None  # first to last gradient-ready event, on the GPU
    cpu_update: float = 0.0            # busiest device worker's kernel time
    stall: float = 0.0                 # time finish_step blocked the caller (includes waiting
                                       # for the GPU, since backward() returns before it finishes)
    per_device_cpu: dict[str, float] = field(default_factory=dict)


class _ComputePool:
    """Update threads for one NUMA node.

    The kernels run on a native thread pool (``max_threads`` threads, the
    caller included); ``parallelism`` caps how many one update uses. A
    one-thread Python executor pinned to the node does first-touch allocation,
    so state is placed on the node whose threads will update it.
    """

    def __init__(self, name: str, cpus: tuple[int, ...] | None, num_threads: int, kernels):
        self.cpus = cpus
        self.max_threads = num_threads
        self.parallelism = num_threads
        self._kernels = kernels
        self.handle = kernels.create_pool(num_threads, cpus) if kernels is not None else None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"adambridge-{name}",
            initializer=hardware.pin_current_thread if cpus else None,
            initargs=(cpus,) if cpus else (),
        )

    def call(self, fn, *args):
        """Run ``fn`` on the pool's allocation thread (NUMA-local first touch)."""
        return self._executor.submit(fn, *args).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
        if self.handle is not None:
            self._kernels.destroy_pool(self.handle)
            self.handle = None


class _Task:
    """Update elements [lo, hi) of ``slot`` with ``grad`` (indexed like the parameter).

    The update may start once ``copied`` (gradient on the host) and, in
    delayed mode, ``written`` (the previous update's ``out`` on the GPU) have
    completed."""

    __slots__ = ("slot", "grad", "copied", "ready", "lo", "hi", "written", "base", "out", "lease")

    def __init__(self, slot, grad, copied=None, ready=None, lo=0, hi=None, written=None, base=0, out=None,
                 lease=None):
        self.slot = slot
        self.grad = grad    # grad[i] is the gradient of element base + i of the parameter
        self.copied = copied
        self.ready = ready
        self.lo = lo
        self.hi = slot.numel if hi is None else hi
        self.written = written
        self.base = base
        self.out = out      # write-back buffer for [lo, hi) in the transfer ring; None: the slot's own ``out``
        self.lease = lease  # the task's lease on the transfer ring

    def landed(self) -> bool:
        return (self.copied is None or self.copied.query()) and (self.written is None or self.written.query())

    def wait_landed(self) -> None:
        if self.copied is not None:
            self.copied.synchronize()
        if self.written is not None:
            self.written.synchronize()

    @property
    def whole(self) -> bool:
        return self.hi - self.lo == self.slot.numel


class _Bundle(list):
    """Tasks queued together (one SmallGroup submission); unpacked by the worker."""


class _DeviceWorker:
    def __init__(self, engine: "UpdateEngine", device: torch.device, pool: _ComputePool):
        self.engine = engine
        self.device = device
        self.pool = pool
        self.queue: queue.SimpleQueue = queue.SimpleQueue()
        self.cuda = device.type == "cuda"
        if self.cuda:
            self.d2h_stream = torch.cuda.Stream(device)
            self.h2d_stream = torch.cuda.Stream(device)
        self.ring: _TransferRing | None = None
        self.busy = 0.0
        self.thread = threading.Thread(target=self._run, name=f"adambridge-worker-{device}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if self.cuda:
            torch.cuda.set_device(self.device)
        if self.pool.cpus:
            hardware.pin_current_thread(self.pool.cpus)
        pending: collections.deque[_Task] = collections.deque()
        stopping = False
        while True:
            if not pending:
                if stopping:
                    return
                item = self.queue.get()
                if item is None:
                    return
                pending.extend(item if isinstance(item, _Bundle) else (item,))
            stopping = self._drain(pending) or stopping
            batch = self._landed(pending)
            try:
                self.engine._process_batch(self, batch)
            except BaseException as exc:  # surfaced by finish_step()
                self.engine._record_error(exc)
            finally:
                self.engine._tasks_done(len(batch), sum(t.lo == 0 for t in batch), batch)

    def _drain(self, pending: collections.deque) -> bool:
        """Move everything queued into ``pending``; True if asked to stop."""
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return False
            if item is None:
                return True
            pending.extend(item if isinstance(item, _Bundle) else (item,))

    @staticmethod
    def _landed(pending: collections.deque) -> list[_Task]:
        """Wait for the oldest copy, then take every task whose copy has also
        completed (copies on one stream complete in order)."""
        first = pending.popleft()
        first.wait_landed()
        batch = [first]
        while pending and pending[0].landed():
            batch.append(pending.popleft())
        return batch

    def stop(self) -> None:
        self.queue.put(None)
        self.thread.join()


class UpdateEngine:
    def __init__(
        self,
        *,
        precision: str,
        backend: str = "auto",
        numa_aware: bool = True,
        num_threads: int | None = None,
        chunk_size: int | None = None,
        stochastic_rounding: bool = True,
        seed: int = 0,
        overlap_writeback: bool = True,
        timing: bool = False,
        algorithm: str = "adamw",
        transfer_buffer_bytes: int | None = None,
        master_precision: str = "fp32",
    ):
        if backend not in ("auto", "native", "torch"):
            raise ValueError("backend must be 'auto', 'native' or 'torch'")
        if algorithm not in native.ALGORITHMS:
            raise ValueError(f"algorithm must be one of {sorted(native.ALGORITHMS)}")
        if master_precision not in native.MASTER_CODES:
            raise ValueError(f"master_precision must be one of {sorted(native.MASTER_CODES)}")
        self.algorithm = algorithm
        self.lion = algorithm == "lion"
        self.delay = False  # see "Delayed updates" in the module docstring; toggle between steps only
        self.native = native.load() if backend in ("auto", "native") else None
        if backend == "native" and self.native is None:
            raise RuntimeError(f"native kernels requested but unavailable: {native.load_error()}")
        self.precision = precision
        self.master_precision = master_precision  # of CUDA and non-FP32 CPU parameters; FP32 CPU ones are their own master
        self.numa_aware = numa_aware
        self.num_threads = num_threads
        self.chunk_size = chunk_size or self.default_chunk_size()
        self.stochastic = stochastic_rounding
        self.seed = seed
        self.overlap_writeback = overlap_writeback
        self.timing = timing

        self._lock = threading.RLock()
        self._cv = threading.Condition(threading.Lock())
        self._pending = 0
        self._errors: list[BaseException] = []
        self._pools: dict[int | None, _ComputePool] = {}
        self._workers: dict[torch.device, _DeviceWorker] = {}
        self._deferred: list[tuple[_DeviceWorker, list[_Task], torch.cuda.Event | None]] = []
        # Delayed mode: slots whose `out` holds an update not yet copied to the
        # parameter, queued in an earlier step (_owed) or in this one (_fresh).
        self._owed: dict[ParamSlot, None] = {}
        self._fresh: dict[ParamSlot, None] = {}
        self._groups: dict[int, tuple[dict, native.Group]] = {}
        self._first_ready = None
        self._last_ready = None
        self._num_updates = 0
        self._closed = False
        self._pinned = _PinnedBuffers()
        # Transfer ring per CUDA device: at most this many bytes (0: one buffer per parameter, as before).
        self.transfer_buffer_bytes = RING_DEFAULT_BYTES if transfer_buffer_bytes is None else transfer_buffer_bytes
        self._transfer_need: dict[torch.device, list[int]] = {}  # device -> [total bytes, largest task bytes]

    def note_transfer(self, param: torch.Tensor) -> None:
        """Count a large CUDA parameter's gradient and write-back bytes towards its device's ring size."""
        n = param.numel()
        per = 2 * param.element_size()
        need = self._transfer_need.setdefault(param.device, [0, 0])
        need[0] += _TransferRing.round_up(n * per) + _RING_ALIGN
        need[1] = max(need[1], _TransferRing.round_up(min(n, _CHUNK) * per) + _RING_ALIGN)

    def _ring_for(self, worker: "_DeviceWorker") -> _TransferRing | None:
        """The device's transfer ring, or None when one buffer per parameter costs no more than the ring."""
        if worker.ring is not None or not self.transfer_buffer_bytes:
            return worker.ring
        total, largest = self._transfer_need.get(worker.device, (0, 0))
        if total <= self.transfer_buffer_bytes:  # (no parameters noted: the engine is used directly)
            return None
        with self._lock:
            if worker.ring is None:
                capacity = max(self.transfer_buffer_bytes, 2 * largest)
                worker.ring = _TransferRing(self._pinned, _TransferRing.round_up(capacity))
        return worker.ring

    def free_buffers(self, slot: ParamSlot) -> None:
        """Drop a slot's host state and transfer buffers (its state moved to the GPU). Between steps only."""
        self.wait()
        torch.cuda.synchronize(slot.param.device)
        self._pinned.free(slot.out)
        self._pinned.free(slot.grad_host)
        slot.master = slot.out = slot.out_view = slot.grad_host = slot.m = slot.v = None
        slot.seg = slot.seg_ptr = None

    def transfer_bytes(self) -> int:
        return sum(w.ring.capacity for w in list(self._workers.values()) if w.ring is not None)

    def _ring_ok(self, slot: ParamSlot, grad: torch.Tensor) -> bool:
        """Can this gradient go through the ring? Only for the native kernel, which reads the gradient and writes
        the weights wherever the task points (the PyTorch fallback indexes the slot's own buffers)."""
        kernels = self.native
        precision = slot.m.precision if slot.m is not None else self.precision
        return (kernels is not None and grad.dtype in (torch.float32, torch.bfloat16)
                and grad.dtype == slot.param.dtype and kernels.supports(grad.dtype, precision, slot.param.dtype))

    def _ensure_out(self, slot: ParamSlot) -> None:
        """Give a CUDA slot its own pinned write-back buffer (delayed updates, or a gradient that could not use
        the ring)."""
        if slot.out is None and slot.param.is_cuda and slot.small is None:
            slot.out = self._pinned.empty(slot.numel, slot.param.dtype)
            slot.out_view = slot.out.view(slot.param.shape)
            slot.seg = None

    @staticmethod
    def default_chunk_size() -> int:
        """Chunk for the PyTorch kernel.

        Each op is split across ATen's intra-op threads, so the chunk's
        working set (~24 B/element) should fit in all cores' L2 together.
        Below ~256K elements per-op dispatch overhead dominates.
        """
        total_l2 = hardware.l2_cache_bytes() * max(1, torch.get_num_threads())
        return max(1 << 18, min(1 << 22, total_l2 // 24 // _ALIGN * _ALIGN))

    # ------------------------------------------------------------------ setup

    def _node_for(self, device: torch.device) -> int | None:
        if not self.numa_aware or device.type != "cuda":
            return None
        node = hardware.gpu_numa_node(device)
        topo = hardware.topology()
        if node is None:
            return next(iter(topo.nodes)) if topo.num_nodes == 1 else None
        return node if node in topo.nodes else None

    def _pool_for(self, node: int | None) -> _ComputePool:
        with self._lock:
            return self._pools.get(node) or self._make_pool(node)

    def _make_pool(self, node: int | None) -> _ComputePool:
        if self._closed:
            raise RuntimeError("UpdateEngine is closed")
        cpus = hardware.topology().cpus(node)
        # Two thirds of the CPUs, at most 32. The update is bandwidth-bound and
        # saturates memory well before every core is busy, while the training
        # loop needs free cores to launch GPU work: with all but two of 24 CPUs
        # updating, Qwen2.5-0.5B's forward and backward ran 25% slower whenever
        # an update overlapped them (delayed updates: 331 vs 280 ms/step with 16).
        threads = self.num_threads or max(1, min(32, len(cpus) * 2 // 3))
        pinned = cpus if (self.numa_aware and node is not None) else None
        name = "all" if node is None else f"node{node}"
        pool = self._pools[node] = _ComputePool(name, pinned, max(1, threads), self.native)
        return pool

    def worker_for(self, device: torch.device) -> _DeviceWorker:
        worker = self._workers.get(device)
        if worker is None:
            with self._lock:
                worker = self._workers.get(device)
                if worker is None:
                    if self._closed:
                        raise RuntimeError("UpdateEngine is closed")
                    pool = self._pool_for(self._node_for(device))
                    worker = self._workers[device] = _DeviceWorker(self, device, pool)
        return worker

    # ------------------------------------------------------------- submission

    def submit_small(self, group: SmallGroup, grads: list) -> list[ParamSlot]:
        """Queue updates for a SmallGroup with one gather and one copy to the
        host. ``grads[i]`` belongs to ``group.slots[i]`` (None: no update this
        step). Returns the slots queued."""
        members = [(slot, g, off) for slot, g, off in zip(group.slots, grads, group.offsets) if g is not None]
        if not members:
            return []
        worker = self.worker_for(group.device)
        slots = [slot for slot, _, _ in members]
        written = None
        if self.delay:
            self._wait_slots(slots)  # the previous update has read the staging buffer and filled `out`
            with self._lock:
                owed = [s for s in slots if s in self._owed]
                for s in slots:
                    self._owed.pop(s, None)
                    self._fresh[s] = None
            if owed:
                written = self._write_back_now(worker, owed, group)
        with self._cv:
            self._pending += len(members)
            for slot in slots:
                slot.inflight += 1
        try:
            bundle = _Bundle()
            stream = worker.d2h_stream
            for dtype in sorted({g.dtype for _, g, _ in members}, key=str):
                part = [(slot, g, off) for slot, g, off in members if g.dtype == dtype]
                staging = group.grad_host(dtype)
                ready = torch.cuda.Event(enable_timing=self.timing)
                copied = torch.cuda.Event()
                whole = len(part) == len(group.slots)
                if whole:  # the usual case: every member, in order, gathered on the GPU
                    flat = torch.cat([g.reshape(-1) for _, g, _ in part])
                ready.record(torch.cuda.current_stream(group.device))
                stream.wait_event(ready)
                with torch.cuda.stream(stream):
                    if whole:
                        staging.copy_(flat, non_blocking=True)
                    else:
                        torch._foreach_copy_([staging[off:off + slot.numel] for slot, _, off in part],
                                             [g.reshape(-1) for _, g, _ in part], non_blocking=True)
                    copied.record(stream)
                if whole:
                    flat.record_stream(stream)
                else:
                    for _, g, _ in part:
                        g.record_stream(stream)
                if self._first_ready is None:
                    self._first_ready = ready
                self._last_ready = ready
                bundle.extend(_Task(slot, staging[off:off + slot.numel], copied, ready, written=written)
                              for slot, _, off in part)
        except BaseException:
            self._tasks_done(len(members), None, slots)
            raise
        worker.queue.put(bundle)
        return slots

    def submit(self, slot: ParamSlot, grad: torch.Tensor) -> None:
        """Queue an update of ``slot`` with ``grad``. Safe to call from autograd threads."""
        worker = self.worker_for(slot.param.device)
        ranges = self._chunks(slot, grad) if worker.cuda else [(0, slot.numel)]
        write_back = False
        if self.delay and worker.cuda:
            self._wait_slots((slot,))  # previous update done: staging free, `out` final
            with self._lock:
                write_back = slot in self._owed
                self._owed.pop(slot, None)
                self._fresh[slot] = None
        with self._cv:
            self._pending += len(ranges)
            slot.inflight += len(ranges)
        ring = self._ring_for(worker) if worker.cuda and self._ring_ok(slot, grad) else None
        if ring is not None:
            # Each task is queued as soon as its copy is: the ring may be smaller than this parameter.
            self._stage_ring(slot, grad, worker, ring, ranges, write_back)
            return
        try:
            if worker.cuda:
                tasks = self._stage_cuda_grad(slot, grad, worker, ranges, write_back)
            else:
                tasks = [_Task(slot, grad.detach().reshape(-1).contiguous())]
        except BaseException:
            self._tasks_done(len(ranges), 0, [slot] * len(ranges))
            raise
        worker.queue.put(tasks[0] if len(tasks) == 1 else _Bundle(tasks))

    def _stage_ring(self, slot: ParamSlot, grad: torch.Tensor, worker: _DeviceWorker, ring: _TransferRing,
                    ranges: list[tuple[int, int]], write_back: bool) -> None:
        """Copy ``grad`` chunk by chunk into ring leases on the D2H stream and queue a task per chunk. Exact
        updates also take their write-back buffer from the lease; delayed ones keep the slot's own ``out``
        (it holds the update until the next step's gradient)."""
        out_in_ring = not self.delay and self.overlap_writeback
        if slot.ready_ev is None or len(slot.copied_evs) != len(ranges):
            slot.ready_ev = torch.cuda.Event(enable_timing=self.timing)
            slot.copied_evs = [torch.cuda.Event() for _ in ranges]
        ready = slot.ready_ev
        ready.record(torch.cuda.current_stream(grad.device))
        written = None
        if write_back:
            h2d = worker.h2d_stream
            h2d.wait_event(ready)
            with torch.cuda.stream(h2d):
                slot.param.data.copy_(slot.out_view, non_blocking=True)
            if slot.written_ev is None:
                slot.written_ev = torch.cuda.Event()
            written = slot.written_ev
            written.record(h2d)
        stream = worker.d2h_stream
        stream.wait_event(ready)
        g_size, o_size = grad.element_size(), slot.param.element_size()
        flat = grad.view(-1) if len(ranges) > 1 else None
        if self._first_ready is None:
            with self._lock:
                if self._first_ready is None:
                    self._first_ready = ready
        self._last_ready = ready
        k = 0
        try:
            for k, ((lo, hi), copied) in enumerate(zip(ranges, slot.copied_evs)):
                n = hi - lo
                g_bytes = _TransferRing.round_up(n * g_size)
                lease = ring.acquire(g_bytes + (n * o_size if out_in_ring else 0))
                g_view = ring.view(lease, grad.dtype, n)
                out = ring.view(lease, slot.param.dtype, n, g_bytes) if out_in_ring else None
                with torch.cuda.stream(stream):
                    if flat is None:
                        g_view.view(grad.shape).copy_(grad, non_blocking=True)
                    else:
                        g_view.copy_(flat[lo:hi], non_blocking=True)
                    copied.record(stream)
                worker.queue.put(_Task(slot, g_view, copied, ready, lo, hi, written, base=lo, out=out, lease=lease))
        except BaseException:
            left = len(ranges) - k
            self._tasks_done(left, 0, [slot] * left)
            raise
        finally:
            grad.record_stream(stream)  # keep the allocation alive until the copies finish

    @staticmethod
    def _chunks(slot: ParamSlot, grad: torch.Tensor) -> list[tuple[int, int]]:
        n = slot.numel
        if n <= _CHUNK or not (grad.is_contiguous() and slot.param.is_contiguous()):
            return [(0, n)]
        return [(lo, min(lo + _CHUNK, n)) for lo in range(0, n, _CHUNK)]

    def _stage_cuda_grad(self, slot: ParamSlot, grad: torch.Tensor, worker: _DeviceWorker,
                         ranges: list[tuple[int, int]], write_back: bool = False) -> list[_Task]:
        """Copy ``grad`` to the slot's pinned staging buffer on the D2H stream.
        With ``write_back`` (delayed mode), first copy the slot's pending
        update to the parameter on the H2D stream; the next update of ``out``
        waits for that copy."""
        staging = slot.grad_host
        if staging is None or staging.dtype != grad.dtype:
            staging = slot.grad_host = self._pinned.empty(slot.numel, grad.dtype)
            slot.seg = None  # re-describe the buffers for the native kernel
        if slot.ready_ev is None or len(slot.copied_evs) != len(ranges):
            slot.ready_ev = torch.cuda.Event(enable_timing=self.timing)
            slot.copied_evs = [torch.cuda.Event() for _ in ranges]
        ready = slot.ready_ev
        ready.record(torch.cuda.current_stream(grad.device))
        written = None
        if write_back:
            h2d = worker.h2d_stream
            h2d.wait_event(ready)
            with torch.cuda.stream(h2d):
                slot.param.data.copy_(slot.out_view, non_blocking=True)
            if slot.written_ev is None:
                slot.written_ev = torch.cuda.Event()
            written = slot.written_ev
            written.record(h2d)
        stream = worker.d2h_stream
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            if len(ranges) == 1:
                staging.view(grad.shape).copy_(grad, non_blocking=True)
                slot.copied_evs[0].record(stream)
            else:
                flat = grad.view(-1)
                for (lo, hi), copied in zip(ranges, slot.copied_evs):
                    staging[lo:hi].copy_(flat[lo:hi], non_blocking=True)
                    copied.record(stream)
        grad.record_stream(stream)  # keep the allocation alive until the copy finishes
        if self._first_ready is None:
            with self._lock:
                if self._first_ready is None:
                    self._first_ready = ready
        self._last_ready = ready
        return [_Task(slot, staging, copied, ready, lo, hi, written) for (lo, hi), copied in zip(ranges, slot.copied_evs)]

    # -------------------------------------------------------------- execution

    def _process_batch(self, worker: _DeviceWorker, batch: list[_Task]) -> None:
        # Every task's gradient copy has completed (see _DeviceWorker._landed).
        # A write-back still in flight reads other slots' `out` buffers: a slot
        # is updated again only after its next gradient copy, which the GPU
        # orders after this step's write-backs (finish_step's wait_stream).
        t0 = time.perf_counter()
        try:
            fresh = list(dict.fromkeys(t.slot for t in batch if not t.slot.initialized))
            if fresh:
                self.initialize_many(fresh, worker.pool)
            for task in batch:
                if task.out is None:  # writes the slot's own buffer: make sure it has one
                    self._ensure_out(task.slot)

            segments, groups, fallback = [], {}, []
            for task in batch:
                slot = task.slot
                if task.lo == 0:
                    slot.step += 1  # a chunked update counts once, at its first chunk
                seg = self._segment(slot, task.grad, task.out)
                if seg is None:
                    if task.out is not None or task.base:
                        raise RuntimeError("internal error: a transfer-ring task needs the native kernel")
                    fallback.append(task)
                    continue
                seg.step = slot.step
                group = slot.group
                entry = groups.get(id(group))
                if entry is None:
                    entry = groups[id(group)] = self._group_struct(group)
                if slot.seg_group is not entry[1]:
                    slot.seg.group = ctypes.pointer(entry[1])
                    slot.seg_group = entry[1]
                segments.append(slot.seg_ptr if task.whole else self._chunk_segment(task))
            if segments:
                self.native.multi_step(worker.pool.handle, self.native.segment_array(segments),
                                       worker.pool.parallelism)
            for task in fallback:
                self._update_torch(task.slot, task.grad, task.lo, task.hi)
            worker.busy += time.perf_counter() - t0
            for task in batch:  # gradient read: leases without a write-back buffer are free again
                if task.lease is not None and task.out is None:
                    worker.ring.release(task.lease)

            if worker.cuda and not self.delay:  # delayed: written back at the next gradient (see submit)
                ready = batch[-1].ready
                if self.overlap_writeback:
                    self._writeback(worker, batch, ready)
                else:
                    with self._lock:
                        self._deferred.append((worker, batch, ready))
        except BaseException:
            for task in batch:
                if task.lease is not None:
                    worker.ring.release(task.lease)  # no copy reads it any more; keep the ring moving
            raise

    @staticmethod
    def _chunk_segment(task: _Task):
        """Pointer to a segment for elements [lo, hi) of the task's slot: the
        slot's buffers advanced to ``lo``, same parameter index and offset ``lo``
        (so the same rounding bits as a whole-tensor update)."""
        slot, lo = task.slot, task.lo
        seg = slot.seg
        m = slot.m.native_buffers[0]
        v = None if slot.v is None else slot.v.native_buffers[0]
        if task.out is not None:
            out = task.out.data_ptr()
        else:
            out = slot.out.data_ptr() + slot.out.element_size() * lo if slot.out is not None else None
        sub = native.Segment(
            p=seg.p + (2 if seg.p_dtype == native.DT_F24 else 4) * lo, p_lo=seg.p_lo + lo if seg.p_lo else None,
            g=task.grad.data_ptr() + task.grad.element_size() * (lo - task.base),
            m=seg.m + m.element_size() * lo, v=seg.v + v.element_size() * lo if v is not None else None,
            m_scale=seg.m_scale + 4 * (lo // FP8_BLOCK) if seg.m_scale else None,
            v_scale=seg.v_scale + 4 * (lo // FP8_BLOCK) if seg.v_scale else None,
            out=out,
            group=seg.group, n=task.hi - lo, step=seg.step, index=seg.index, offset=lo,
            g_dtype=seg.g_dtype, mv_dtype=seg.mv_dtype, out_dtype=seg.out_dtype, algorithm=seg.algorithm,
            p_dtype=seg.p_dtype,
        )
        return ctypes.pointer(sub)

    def _group_struct(self, group: dict) -> tuple[dict, native.Group]:
        """The native copy of a parameter group's hyperparameters, refreshed
        from the dict (an LR scheduler may have changed it since last step)."""
        entry = self._groups.get(id(group))
        if entry is None or entry[0] is not group:
            entry = self._groups[id(group)] = (group, native.Group())
        g = entry[1]
        beta1, beta2 = group["betas"]
        g.lr = float(group["lr"])
        g.beta1 = float(beta1)
        g.beta2 = float(beta2)
        g.eps = float(group.get("eps", 0.0))  # Lion has none
        g.weight_decay = float(group["weight_decay"])
        g.seed = self.seed & _MASK64
        g.stochastic = int(self.stochastic)
        return entry

    def _segment(self, slot: ParamSlot, grad: torch.Tensor, out: torch.Tensor | None = None) -> native.Segment | None:
        """The slot's native segment pointing at ``grad`` and writing the new
        weights to ``out`` (a transfer-ring buffer) or the slot's own ``out``,
        or None if the native kernel cannot update this slot (it then takes
        the PyTorch path)."""
        kernels = self.native
        if kernels is None:
            return None
        seg = slot.seg
        if seg is None:
            out_dtype = (slot.out.dtype if slot.out is not None
                         else slot.param.dtype if slot.param.is_cuda and slot.small is None else None)
            m_buf = slot.m.native_buffers or (None, None)
            v_buf = (None, None) if slot.v is None else (slot.v.native_buffers or (None, None))
            if (m_buf[0] is None or (v_buf[0] is None and not self.lion)
                    or not kernels.supports(grad.dtype, slot.m.precision, out_dtype)):
                return None
            seg = native.Segment()
            p_hi, p_lo = slot.master.native_buffers
            seg.p = p_hi.data_ptr()
            seg.p_lo = p_lo.data_ptr() if p_lo is not None else None
            seg.p_dtype = native.MASTER_CODES[slot.master.precision]
            seg.m = m_buf[0].data_ptr()
            seg.v = v_buf[0].data_ptr() if v_buf[0] is not None else None
            seg.m_scale = m_buf[1].data_ptr() if m_buf[1] is not None else None
            seg.v_scale = v_buf[1].data_ptr() if v_buf[1] is not None else None
            seg.out = slot.out.data_ptr() if slot.out is not None else None
            seg.n = slot.numel
            seg.index = slot.index
            seg.mv_dtype = native.MOMENT_CODES[slot.m.precision]
            seg.out_dtype = kernels.dtype_code(out_dtype)
            seg.algorithm = native.ALGORITHMS[self.algorithm]
            slot.seg, slot.seg_ptr, slot.seg_group = seg, ctypes.pointer(seg), None
        if grad.dtype not in (torch.float32, torch.bfloat16):
            return None
        seg.g = grad.data_ptr()  # staging buffer (CUDA) or this step's gradient (CPU)
        seg.g_dtype = kernels.dtype_code(grad.dtype)
        if out is not None:
            seg.out = out.data_ptr()
        elif slot.out is not None:
            seg.out = slot.out.data_ptr()
        return seg

    def initialize_many(self, slots: list[ParamSlot], pool: _ComputePool | None = None) -> None:
        """Allocate master weights and moments, first-touched on the pool's node."""
        precision = self.precision

        def alloc(p, numel, small):
            src = p.detach().reshape(-1)
            if p.device.type == "cpu" and p.dtype == torch.float32 and p.is_contiguous():
                master, out = FP32Store.wrap(p.data.view(-1)), None  # update the parameter in place
            elif p.device.type == "cpu":
                if not p.is_contiguous():
                    raise ValueError("non-contiguous CPU parameters are not supported")
                master, out = make_master(self.master_precision, numel, init=src), p.data.view(-1)
            else:
                master = make_master(self.master_precision, numel, init=src)
                # Large CUDA parameters write their exact updates into the transfer ring when the device has one;
                # delayed updates (and a ring that is off) need a buffer of their own (see _ensure_out).
                ring = (not small and self.transfer_buffer_bytes and not self.delay and self.overlap_writeback
                        and self.native is not None and p.dtype in (torch.float32, torch.bfloat16)
                        and self.native.supports(p.dtype, precision, p.dtype)
                        and sum(self._transfer_need.get(p.device, (0, 0))[:1]) > self.transfer_buffer_bytes)
                out = None if small or ring else self._pinned.empty(numel, p.dtype)
            v = None if self.lion else make_store(precision, numel, "v")
            return master, out, make_store(precision, numel, "m"), v

        def alloc_all():
            return [alloc(s.param, s.numel, s.small is not None) for s in slots]

        if pool is None:
            pool = self._pool_for(self._node_for(slots[0].param.device))
        for slot, (master, out, m, v) in zip(slots, pool.call(alloc_all)):
            if slot.small is not None:
                out = slot.small.out_for(slot)
            slot.master, slot.out, slot.m, slot.v = master, out, m, v
            slot.out_view = out.view(slot.param.shape) if out is not None else None
            slot.seg = None

    def initialize(self, slot: ParamSlot, pool: _ComputePool | None = None) -> None:
        self.initialize_many([slot], pool)

    def convert_moments(self, slot: ParamSlot, precision: str) -> None:
        """Re-encode an initialized slot's moments in another precision."""
        pool = self._pool_for(self._node_for(slot.param.device))

        def convert():
            return (make_store(precision, slot.numel, "m", init=slot.m.to_fp32()),
                    None if slot.v is None else make_store(precision, slot.numel, "v", init=slot.v.to_fp32()))

        slot.m, slot.v = pool.call(convert)
        slot.seg = None

    def replace_moments(self, slot: ParamSlot, m: MomentStore, v: MomentStore | None) -> None:
        slot.m, slot.v = m, v
        slot.seg = None

    def _update_torch(self, slot: ParamSlot, grad: torch.Tensor, lo: int = 0, hi: int | None = None) -> None:
        group = slot.group
        beta1, beta2 = group["betas"]
        seed = mix_seed(self.seed, slot.index, slot.step)
        rng_seed = seed if self.stochastic else None
        if self.lion:
            c = StepCoefficients.for_lion(float(group["lr"]), beta1, beta2, group["weight_decay"])
            for a, b in chunk_ranges(lo, slot.numel if hi is None else hi, self.chunk_size):
                fused_lion_chunk(slot.master, grad, slot.m, slot.out, c, a, b, seed=rng_seed)
            return
        c = StepCoefficients.for_step(
            float(group["lr"]), beta1, beta2, group["eps"], group["weight_decay"], slot.step
        )
        for a, b in chunk_ranges(lo, slot.numel if hi is None else hi, self.chunk_size):
            fused_adamw_chunk(slot.master, grad, slot.m, slot.v, slot.out, c, a, b, seed=rng_seed)

    def uses_native(self, grad_dtype: torch.dtype, precision: str, out_dtype: torch.dtype | None) -> bool:
        return self.native is not None and self.native.supports(grad_dtype, precision, out_dtype)

    def run_kernel(self, master, grad, m, v, out, c: StepCoefficients, seed: int, pool: _ComputePool) -> None:
        """One standalone update with explicit coefficients (benchmarks, tests).
        ``master`` is a flat FP32 tensor or a master store; ``v=None`` runs Lion
        (``c`` from ``StepCoefficients.for_lion``)."""
        if isinstance(master, torch.Tensor):
            n, p_hi, p_lo, p_code = master.numel(), master, None, native.DT_F32
        else:
            n, (p_hi, p_lo), p_code = master.numel, master.native_buffers, native.MASTER_CODES[master.precision]
        if n == 0:
            return
        lion = v is None
        out_dtype = None if out is None else out.dtype
        if (self.uses_native(grad.dtype, m.precision, out_dtype)
                and m.native_buffers is not None and (lion or v.native_buffers is not None)):
            kernels = self.native
            m_data, m_scale = m.native_buffers
            v_data, v_scale = (None, None) if lion else v.native_buffers
            seg = native.Segment(
                p=p_hi.data_ptr(), p_lo=p_lo.data_ptr() if p_lo is not None else None, p_dtype=p_code,
                g=grad.data_ptr(), m=m_data.data_ptr(),
                v=v_data.data_ptr() if v_data is not None else None,
                m_scale=m_scale.data_ptr() if m_scale is not None else None,
                v_scale=v_scale.data_ptr() if v_scale is not None else None,
                out=out.data_ptr() if out is not None else None,
                n=n, g_dtype=kernels.dtype_code(grad.dtype), mv_dtype=native.MOMENT_CODES[m.precision],
                out_dtype=kernels.dtype_code(out_dtype), algorithm=native.ALGORITHMS["lion" if lion else "adamw"],
            )
            hp = kernels.make_hparams(c, seed, self.stochastic)
            kernels.multi_apply(pool.handle, kernels.segment_array([ctypes.pointer(seg)]), [hp], pool.parallelism)
            return
        rng_seed = seed if self.stochastic else None
        for lo, hi in chunk_ranges(0, n, self.chunk_size):
            if lion:
                fused_lion_chunk(master, grad, m, out, c, lo, hi, seed=rng_seed)
            else:
                fused_adamw_chunk(master, grad, m, v, out, c, lo, hi, seed=rng_seed)

    def _writeback(self, worker: _DeviceWorker, tasks: list[_Task], ready) -> None:
        stream = worker.h2d_stream
        if ready is not None:
            stream.wait_event(ready)
        dst, src, groups = [], [], {}
        for task in tasks:
            slot = task.slot
            if slot.small is not None:
                groups.setdefault(id(slot.small), (slot.small, []))[1].append(slot)
            elif task.whole:
                dst.append(slot.param.data)
                src.append(task.out.view(slot.param.shape) if task.out is not None else slot.out_view)
            else:
                dst.append(slot.param.data.view(-1)[task.lo:task.hi])
                src.append(task.out if task.out is not None else slot.out[task.lo:task.hi])
        leases = [t.lease for t in tasks if t.lease is not None and t.out is not None]
        with torch.cuda.stream(stream):
            for group, members in groups.values():
                if len(members) == len(group.slots):
                    group.write_back()      # one copy + one fused scatter
                else:
                    dst.extend(s.param.data for s in members)
                    src.extend(s.out_view for s in members)
            if dst:
                torch._foreach_copy_(dst, src, non_blocking=True)
            if leases:
                copied = torch.cuda.Event()
                copied.record(stream)
        for lease in leases:  # free once the weights are on the GPU
            worker.ring.release(lease, copied)

    # ---------------------------------------------------------- synchronizing

    def _tasks_done(self, count: int, updates: int | None = None, done=()) -> None:
        """``count`` tasks finished, completing ``updates`` parameter updates
        (chunks count once). ``done`` holds each finished task, or its slot."""
        with self._cv:
            self._pending -= count
            self._num_updates += count if updates is None else updates
            for item in done:
                (item.slot if isinstance(item, _Task) else item).inflight -= 1
            self._cv.notify_all()

    def _wait_slots(self, slots) -> None:
        """Block until no update of ``slots`` is queued or running."""
        with self._cv:
            while any(s.inflight for s in slots):
                self._cv.wait()

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._errors.append(exc)

    def _raise_errors(self) -> None:
        if self._errors:
            with self._lock:
                errors, self._errors = self._errors, []
            raise RuntimeError(f"adambridge host update failed ({len(errors)} error(s))") from errors[0]

    def wait(self) -> None:
        """Block until every submitted update has run on the host."""
        with self._cv:
            while self._pending:
                self._cv.wait()
        self._raise_errors()

    def _write_back_now(self, worker: _DeviceWorker, slots: list[ParamSlot],
                        group: SmallGroup | None = None) -> torch.cuda.Event:
        """Delayed mode: copy ``slots``' finished updates to their parameters on
        the H2D stream, after the GPU work queued so far. Returns the copy's
        completion event."""
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(worker.device))
        stream = worker.h2d_stream
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            if group is not None and len(slots) == len(group.slots):
                group.write_back()  # one copy + one fused scatter
            else:
                torch._foreach_copy_([s.param.data for s in slots], [s.out_view for s in slots], non_blocking=True)
        done = torch.cuda.Event()
        done.record(stream)
        return done

    def _write_back_slots(self, slots: list[ParamSlot]) -> None:
        """Wait for ``slots``' updates and copy them to their parameters."""
        if not slots:
            return
        self._wait_slots(slots)
        by_device: dict[torch.device, list[ParamSlot]] = {}
        for slot in slots:
            by_device.setdefault(slot.param.device, []).append(slot)
        for device, members in by_device.items():
            self._write_back_now(self._workers[device], members)

    def flush(self) -> None:
        """Delayed mode: finish every queued update and copy all of them to the
        parameters, so the next GPU work on the current stream sees the
        weights an exact (undelayed) optimizer would have produced."""
        self.wait()
        with self._lock:
            slots = list({**self._owed, **self._fresh})
            self._owed, self._fresh = {}, {}
        self._write_back_slots(slots)
        self._order_writebacks()

    def discard_pending(self) -> None:
        """Wait for queued updates and drop any not yet written to the
        parameters (the host state is about to be replaced)."""
        self.wait()
        with self._lock:
            self._owed, self._fresh = {}, {}

    def _order_writebacks(self) -> None:
        for worker in list(self._workers.values()):
            if worker.cuda:
                torch.cuda.current_stream(worker.device).wait_stream(worker.h2d_stream)

    def finish_step(self) -> StepStats:
        """Wait for host updates, order write-backs before the caller's next GPU work.

        Delayed mode: no waiting. Updates from earlier steps whose parameters
        got no gradient this step are written back now (so no parameter lags by
        more than one step); this step's updates are written back by the next
        step's gradients."""
        t0 = time.perf_counter()
        if self.delay and all(w.cuda for w in list(self._workers.values())):
            self._raise_errors()
        else:  # CPU parameters are updated in place: never delayed
            self.wait()
        if self.delay:
            with self._lock:
                left, self._owed, self._fresh = list(self._owed), self._fresh, {}
            self._write_back_slots(left)
        with self._lock:
            deferred, self._deferred = self._deferred, []
        for worker, tasks, ready in deferred:
            self._writeback(worker, tasks, ready)
        self._order_writebacks()

        stats = StepStats(stall=time.perf_counter() - t0)
        with self._lock:
            first, last = self._first_ready, self._last_ready
            self._first_ready = self._last_ready = None
            stats.num_updates, self._num_updates = self._num_updates, 0
        if self.timing and first is not None and last is not None and first is not last:
            last.synchronize()
            stats.gpu_backward = first.elapsed_time(last) / 1e3
        for worker in self._workers.values():
            stats.per_device_cpu[str(worker.device)] = worker.busy
            stats.cpu_update = max(stats.cpu_update, worker.busy)
            worker.busy = 0.0
        return stats

    # ----------------------------------------------------------------- tuning

    def set_parallelism(self, threads: int) -> None:
        for pool in self._pools.values():
            pool.parallelism = max(1, min(threads, pool.max_threads))

    def pools(self) -> list[_ComputePool]:
        return list(self._pools.values())

    def ensure_pool(self, device: torch.device) -> _ComputePool:
        return self.worker_for(device).pool

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers, self._workers = list(self._workers.values()), {}
            pools, self._pools = list(self._pools.values()), {}
        for worker in workers:
            worker.stop()
        for pool in pools:
            pool.shutdown()
        self._pinned.release()
