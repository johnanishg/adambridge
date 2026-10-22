"""Host update engine: the layer-wise asynchronous offload pipeline.

For a parameter on a CUDA device, one update flows through::

    autograd thread          D2H stream            worker thread            H2D stream
    ---------------          ----------            -------------            ----------
    grad ready (hook)
    record `ready` ------->  wait `ready`
                             grad -> pinned host
                             record `copied` ----> wait `copied` (CPU)
                                                   fused AdamW on host
                                                   (NUMA-local thread pool)
                                                   ------------------------> wait `ready`
                                                                             pinned -> param
    ...backward continues on the compute stream for earlier layers...

``finish_step`` waits for every queued update and makes the caller's current
stream wait on the H2D streams, so the next forward pass sees new weights.

There is one worker thread per device: layers are updated in the order their
gradients become ready, each spread across the device's compute pool. Pools
are per NUMA node and their threads are pinned to that node's CPUs, so
optimizer state first-touched by them is allocated node-locally.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from . import hardware, native
from .kernels import StepCoefficients, chunk_ranges, fused_adamw_chunk, mix_seed, split_range
from .precision import MomentStore, make_store

__all__ = ["UpdateEngine", "ParamSlot", "StepStats"]

log = logging.getLogger("adamx")

# Ranges handed to threads start on this boundary: a multiple of the SIMD
# width (so vector/tail assignment is independent of the split) and of the
# FP8 block size.
_ALIGN = 128


class ParamSlot:
    """Host-side optimizer state for one parameter."""

    __slots__ = (
        "index", "param", "group", "numel", "step",
        "master", "out", "m", "v",
        "grad_host", "h2d_event", "submitted",
    )

    def __init__(self, index: int, param: torch.Tensor, group: dict):
        self.index = index
        self.param = param
        self.group = group
        self.numel = param.numel()
        self.step = 0
        self.master: torch.Tensor | None = None   # FP32, flat, host
        self.out: torch.Tensor | None = None      # write-back buffer in param dtype (pinned for CUDA)
        self.m: MomentStore | None = None
        self.v: MomentStore | None = None
        self.grad_host: torch.Tensor | None = None  # pinned gradient staging (CUDA params)
        self.h2d_event: torch.cuda.Event | None = None
        self.submitted = False

    @property
    def initialized(self) -> bool:
        return self.master is not None


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
    """Threads (optionally pinned to one NUMA node) that run kernel ranges."""

    def __init__(self, name: str, cpus: tuple[int, ...] | None, num_threads: int):
        self.cpus = cpus
        self.max_threads = num_threads
        self.parallelism = num_threads
        self._executor = ThreadPoolExecutor(
            max_workers=num_threads,
            thread_name_prefix=f"adamx-{name}",
            initializer=hardware.pin_current_thread if cpus else None,
            initargs=(cpus,) if cpus else (),
        )

    def run(self, fn, ranges: list[tuple[int, int]]) -> None:
        if len(ranges) == 1:
            fn(*ranges[0])
            return
        futures = [self._executor.submit(fn, lo, hi) for lo, hi in ranges]
        for f in futures:
            f.result()

    def call(self, fn, *args):
        """Run ``fn`` on a pool thread (used for NUMA-local first-touch allocation)."""
        return self._executor.submit(fn, *args).result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


class _Task:
    __slots__ = ("slot", "grad", "copied", "ready")

    def __init__(self, slot, grad, copied=None, ready=None):
        self.slot = slot
        self.grad = grad
        self.copied = copied
        self.ready = ready


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
        self.busy = 0.0
        self.thread = threading.Thread(target=self._run, name=f"adamx-worker-{device}", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if self.cuda:
            torch.cuda.set_device(self.device)
        if self.pool.cpus:
            hardware.pin_current_thread(self.pool.cpus)
        while True:
            task = self.queue.get()
            if task is None:
                return
            try:
                self.engine._process(self, task)
            except BaseException as exc:  # surfaced by finish_step()
                self.engine._record_error(exc)
            finally:
                self.engine._task_done()

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
    ):
        if backend not in ("auto", "native", "torch"):
            raise ValueError("backend must be 'auto', 'native' or 'torch'")
        self.native = native.load() if backend in ("auto", "native") else None
        if backend == "native" and self.native is None:
            raise RuntimeError(f"native kernels requested but unavailable: {native.load_error()}")
        self.precision = precision
        self.numa_aware = numa_aware
        self.num_threads = num_threads
        self.chunk_size = chunk_size or self.default_chunk_size()
        self.stochastic = stochastic_rounding
        self.seed = seed
        self.overlap_writeback = overlap_writeback
        self.timing = timing

        self._lock = threading.RLock()
        self._cv = threading.Condition()
        self._pending = 0
        self._errors: list[BaseException] = []
        self._pools: dict[int | None, _ComputePool] = {}
        self._workers: dict[torch.device, _DeviceWorker] = {}
        self._deferred: list[tuple[ParamSlot, _DeviceWorker, torch.cuda.Event | None]] = []
        self._first_ready = None
        self._last_ready = None
        self._num_updates = 0
        self._closed = False

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
        threads = self.num_threads or len(cpus)
        pinned = cpus if (self.numa_aware and node is not None) else None
        name = "all" if node is None else f"node{node}"
        pool = self._pools[node] = _ComputePool(name, pinned, max(1, threads))
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

    def submit(self, slot: ParamSlot, grad: torch.Tensor) -> None:
        """Queue an update of ``slot`` with ``grad``. Safe to call from autograd threads."""
        worker = self.worker_for(slot.param.device)
        with self._cv:
            self._pending += 1
        try:
            if worker.cuda:
                task = self._stage_cuda_grad(slot, grad, worker)
            else:
                task = _Task(slot, grad.detach().reshape(-1).contiguous())
        except BaseException:
            self._task_done()
            raise
        worker.queue.put(task)

    def _stage_cuda_grad(self, slot: ParamSlot, grad: torch.Tensor, worker: _DeviceWorker) -> _Task:
        if slot.grad_host is None or slot.grad_host.dtype != grad.dtype:
            slot.grad_host = torch.empty(slot.numel, dtype=grad.dtype, pin_memory=True)
        ready = torch.cuda.Event(enable_timing=self.timing)
        ready.record(torch.cuda.current_stream(grad.device))
        stream = worker.d2h_stream
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            slot.grad_host.view(grad.shape).copy_(grad, non_blocking=True)
            copied = torch.cuda.Event()
            copied.record(stream)
        grad.record_stream(stream)  # keep the allocation alive until the copy finishes
        with self._lock:
            if self._first_ready is None:
                self._first_ready = ready
            self._last_ready = ready
        return _Task(slot, slot.grad_host, copied, ready)

    # -------------------------------------------------------------- execution

    def _process(self, worker: _DeviceWorker, task: _Task) -> None:
        slot = task.slot
        if task.copied is not None:
            task.copied.synchronize()
        t0 = time.perf_counter()
        if not slot.initialized:
            self.initialize(slot, worker.pool)
        if slot.h2d_event is not None:
            slot.h2d_event.synchronize()  # last step's write-back must not see this step's values half-written
        self._update(slot, task.grad, worker.pool)
        worker.busy += time.perf_counter() - t0
        if worker.cuda:
            if self.overlap_writeback:
                self._writeback(slot, worker, task.ready)
            else:
                with self._lock:
                    self._deferred.append((slot, worker, task.ready))

    def initialize(self, slot: ParamSlot, pool: _ComputePool | None = None) -> None:
        """Allocate master weights and moments, first-touched by the node's threads."""
        p = slot.param

        def alloc():
            src = p.detach().reshape(-1)
            if p.device.type == "cpu" and p.dtype == torch.float32 and p.is_contiguous():
                master, out = p.data.view(-1), None  # update the parameter in place
            elif p.device.type == "cpu":
                if not p.is_contiguous():
                    raise ValueError("non-contiguous CPU parameters are not supported")
                master, out = src.float().clone(), p.data.view(-1)
            else:
                master = torch.empty(slot.numel, dtype=torch.float32)
                master.copy_(src)
                out = torch.empty(slot.numel, dtype=p.dtype, pin_memory=True)
            return master, out, make_store(self.precision, slot.numel, "m"), make_store(self.precision, slot.numel, "v")

        if pool is None:
            pool = self._pool_for(self._node_for(p.device))
        slot.master, slot.out, slot.m, slot.v = pool.call(alloc)

    def convert_moments(self, slot: ParamSlot, precision: str) -> None:
        """Re-encode an initialized slot's moments in another precision."""
        pool = self._pool_for(self._node_for(slot.param.device))

        def convert():
            return (make_store(precision, slot.numel, "m", init=slot.m.to_fp32()),
                    make_store(precision, slot.numel, "v", init=slot.v.to_fp32()))

        slot.m, slot.v = pool.call(convert)

    def _update(self, slot: ParamSlot, grad: torch.Tensor, pool: _ComputePool) -> None:
        slot.step += 1
        group = slot.group
        beta1, beta2 = group["betas"]
        c = StepCoefficients.for_step(
            float(group["lr"]), beta1, beta2, group["eps"], group["weight_decay"], slot.step
        )
        seed = mix_seed(self.seed, slot.index, slot.step)
        self.run_kernel(slot.master, grad, slot.m, slot.v, slot.out, c, seed, pool)

    def uses_native(self, grad_dtype: torch.dtype, precision: str, out_dtype: torch.dtype | None) -> bool:
        return self.native is not None and self.native.supports(grad_dtype, precision, out_dtype)

    def run_kernel(self, master, grad, m, v, out, c: StepCoefficients, seed: int, pool: _ComputePool) -> None:
        """Update every element. Native: ranges split across the pool's pinned
        threads. PyTorch: chunk by chunk on this thread, each op parallelized by ATen."""
        n = master.numel()
        if n == 0:
            return
        if self.uses_native(grad.dtype, m.precision, None if out is None else out.dtype):
            kernels = self.native
            hp = kernels.make_hparams(c, seed, self.stochastic)
            m_raw, v_raw = m.raw, v.raw
            pool.run(lambda lo, hi: kernels.fused_adamw(master, grad, m_raw, v_raw, out, hp, lo, hi),
                     split_range(n, pool.parallelism, _ALIGN))
            return
        rng_seed = seed if self.stochastic else None
        for lo, hi in chunk_ranges(0, n, self.chunk_size):
            fused_adamw_chunk(master, grad, m, v, out, c, lo, hi, seed=rng_seed)

    def _writeback(self, slot: ParamSlot, worker: _DeviceWorker, ready) -> None:
        stream = worker.h2d_stream
        if ready is not None:
            stream.wait_event(ready)
        with torch.cuda.stream(stream):
            slot.param.data.copy_(slot.out.view(slot.param.shape), non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        slot.h2d_event = event

    # ---------------------------------------------------------- synchronizing

    def _task_done(self) -> None:
        with self._cv:
            self._pending -= 1
            self._num_updates += 1
            if self._pending == 0:
                self._cv.notify_all()

    def _record_error(self, exc: BaseException) -> None:
        with self._lock:
            self._errors.append(exc)

    def wait(self) -> None:
        """Block until every submitted update has run on the host."""
        with self._cv:
            while self._pending:
                self._cv.wait()
        if self._errors:
            errors, self._errors = self._errors, []
            raise RuntimeError(f"adamx host update failed ({len(errors)} error(s))") from errors[0]

    def finish_step(self) -> StepStats:
        """Wait for host updates, order write-backs before the caller's next GPU work."""
        t0 = time.perf_counter()
        self.wait()
        with self._lock:
            deferred, self._deferred = self._deferred, []
        for slot, worker, ready in deferred:
            self._writeback(slot, worker, ready)
        for worker in list(self._workers.values()):
            if worker.cuda:
                torch.cuda.current_stream(worker.device).wait_stream(worker.h2d_stream)

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
