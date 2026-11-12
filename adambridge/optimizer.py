"""AdamBridge: AdamW with optimizer state offloaded to host memory and updated by
SIMD kernels on the CPU, overlapped with the GPU backward pass. LionBridge: the
same for Lion."""

from __future__ import annotations

import contextlib
import logging
import weakref
from typing import Any, Callable, Iterable

import torch

from . import autotune as _autotune
from .engine import ParamSlot, SmallGroup, StepStats, UpdateEngine
from .precision import (FP32Store, _checked, bytes_per_param, canonical_master_precision, canonical_precision,
                        check_master_state, load_master_, make_store, master_to_fp32, store_from_state_dict)

__all__ = ["AdamBridge", "LionBridge", "OptiComputeAdamW"]

log = logging.getLogger("adambridge")


class AdamBridge(torch.optim.Optimizer):
    r"""AdamW whose state lives in (NUMA-local) host memory.

    Numerically this is ``torch.optim.AdamW`` (decoupled weight decay, bias
    correction, same operation order). FP32 master weights and both moments
    are kept on the host; for CUDA parameters only the parameters themselves
    (in their own dtype) occupy GPU memory.

    Two ways to drive it:

    **Synchronous** - a drop-in replacement for any optimizer::

        loss.backward()
        opt.step()          # gradients -> host, fused update, weights -> GPU
        opt.zero_grad()

    **Layer-wise asynchronous** - updates overlap the backward pass::

        opt.attach_hooks(model)
        loss.backward()     # each parameter is offloaded and updated as soon as its gradient is ready
        opt.step_sync()     # wait; the next forward pass sees the new weights

    In asynchronous mode gradients are released from the GPU as soon as they
    are copied out (``release_grads=True``), so ``p.grad`` is ``None`` after
    ``backward()``. Global gradient-norm clipping is not possible in this mode
    since the first parameters are updated before the last gradients exist;
    use synchronous mode (clip before ``step()``) if you need it. For gradient
    accumulation, run all but the last micro-batch inside :meth:`no_sync`, or
    pass ``accumulation_steps`` (needed when a training loop you don't
    control, such as the Hugging Face ``Trainer``, runs the micro-batches).

    **Delayed updates** (``delay_updates=True``) remove the optimizer from the
    critical path: :meth:`step_sync` returns at once and each parameter's
    update is applied during the *next* backward pass, so the host update
    overlaps a whole forward and backward pass. The model then trains with
    weights one step stale (as DeepSpeed ZeRO-Offload's delayed parameter
    update); ``delay_warmup_steps`` keeps the first steps exact. Call
    :meth:`flush` before evaluating or saving the model, to apply the pending
    updates (``state_dict()`` and ``close()`` do so themselves).

    Args:
        params: iterable of parameters or parameter groups.
        lr, betas, eps, weight_decay: as in ``torch.optim.AdamW``.
        precision: moment storage: ``'fp32'``, ``'bf16'`` (alias
            ``'bf16_moments'``) or ``'fp8'`` (alias ``'fp8_moments'``; E4M3 for
            ``m``, E5M2 for ``sqrt(v)``, per-128-element scales).
        master_precision: ``'fp32'`` (default) or ``'fp24'``: the master weights
            kept as the top 24 bits of their FP32 value (3 bytes per parameter
            instead of 4, relative precision 2^-16, updates rounded
            stochastically). FP32 CPU parameters are their own master either way.
        stochastic_rounding: round low-precision moments (and an FP24 master)
            stochastically. Disabling it makes BF16/FP8 moments stall for
            ``beta2`` near 1.
        backend: ``'auto'`` (native SIMD kernel when it can be built, else
            PyTorch), ``'native'`` or ``'torch'``.
        numa_aware: allocate state and pin update threads on the NUMA node
            each GPU is attached to (Linux).
        num_threads: update threads per NUMA node (default: two thirds of its
            CPUs, at most 32; the rest keep the training loop feeding the GPU).
        chunk_size: elements per PyTorch-kernel chunk (default from L2 size).
        release_grads: in asynchronous mode, free each gradient once it has
            been copied to the host.
        overlap_writeback: copy updated weights to the GPU as soon as each is
            ready. If False, all copies are issued in :meth:`step_sync`. Turn
            it off if some op reads a parameter during backward *after* that
            parameter's gradient is complete (only possible via ``.detach()``
            tricks).
        autotune: ``True`` or an :class:`~adambridge.autotune.AutoTuneConfig` to
            tune thread count and moment precision at runtime.
        seed: base seed for stochastic rounding.
        small_param_threshold: CUDA parameters with fewer elements (norms,
            biases) get no per-tensor hook; they are updated together at
            :meth:`step_sync` with one transfer each way. Their update takes
            microseconds, so overlapping it with backward gains nothing, while a
            hook per tensor costs tens of microseconds of Python each.
        delay_updates: apply each update one step late, overlapping it with
            the next forward and backward pass (CUDA parameters only).
        delay_warmup_steps: with ``delay_updates``, run this many exact steps
            first.
        accumulation_steps: in asynchronous mode, update only on every N-th
            backward pass (the gradients of the others accumulate). ``step()``
            or ``step_sync()`` resets the count.
        transfer_buffer_mb: pinned host memory per GPU shared by the large
            parameters' gradient copies and weight write-backs (a ring). By
            default 512 MB, used only when one buffer per parameter would take
            more (models above ~128M BF16 parameters); 0 keeps one buffer per
            parameter (4 bytes per BF16 parameter for the whole model). Delayed
            updates keep a write-back buffer per parameter either way.
        gpu_state_budget: keep the optimizer state of some CUDA parameters on
            the GPU instead (FP32 master and moments, 12 bytes per parameter,
            updated by fused AdamW at :meth:`step_sync`). ``"auto"``: after the
            first step, as many as the GPU's spare memory holds (keeping a
            margin of 1 GiB or 5%, whichever is larger); an int: at most that
            many bytes; ``None`` or 0: none. The parameters chosen are the last
            to receive their gradients in backward: their host update cannot
            overlap the backward pass, so it is what the step waits for. When
            the whole model fits, this is all-GPU AdamW; when nothing fits, it
            is the default offload. AdamW only.
    """

    _algorithm = "adamw"

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        precision: str = "fp32",
        stochastic_rounding: bool = True,
        backend: str = "auto",
        numa_aware: bool = True,
        num_threads: int | None = None,
        chunk_size: int | None = None,
        release_grads: bool = True,
        overlap_writeback: bool = True,
        autotune: bool | _autotune.AutoTuneConfig = False,
        seed: int = 0,
        small_param_threshold: int = 65536,
        delay_updates: bool = False,
        delay_warmup_steps: int = 0,
        accumulation_steps: int = 1,
        transfer_buffer_mb: int | None = None,
        gpu_state_budget: int | str | None = None,
        master_precision: str = "fp32",
    ):
        if not 0.0 <= lr:
            raise ValueError(f"invalid learning rate: {lr}")
        if delay_warmup_steps < 0 or accumulation_steps < 1:
            raise ValueError("delay_warmup_steps must be >= 0 and accumulation_steps >= 1")
        if not 0.0 <= eps:
            raise ValueError(f"invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if chunk_size is not None and (chunk_size <= 0 or chunk_size % 128):
            raise ValueError("chunk_size must be a positive multiple of 128")
        if transfer_buffer_mb is not None and transfer_buffer_mb < 0:
            raise ValueError("transfer_buffer_mb must be >= 0")
        if not (gpu_state_budget is None or gpu_state_budget == "auto"
                or (isinstance(gpu_state_budget, int) and gpu_state_budget >= 0)):
            raise ValueError("gpu_state_budget must be None, 'auto' or a number of bytes >= 0")
        if gpu_state_budget and self._algorithm != "adamw":
            raise ValueError("gpu_state_budget is only available for AdamW")
        self.gpu_state_budget = gpu_state_budget or None
        self._grad_order: list[ParamSlot] = []  # first step: the order in which gradients arrived
        self._gpu_opts: list[tuple[dict, torch.optim.AdamW, list[ParamSlot]]] = []
        self._gpu_placed = False

        self.precision = canonical_precision(precision)
        self.master_precision = canonical_master_precision(master_precision)
        self.release_grads = release_grads
        self.small_param_threshold = small_param_threshold
        self._slots: dict[torch.Tensor, ParamSlot] = {}
        self._ordered: list[ParamSlot] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._hooked = False
        self._accumulating = 0
        self.accumulation_steps = accumulation_steps
        self._micro = 0                  # backward passes since the last step
        self._backward_counted = False   # an end-of-backward callback is queued
        self.delay_updates = delay_updates
        self.delay_warmup_steps = delay_warmup_steps
        self._steps_synced = 0
        self.last_step_stats: StepStats | None = None

        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

        if autotune is True:
            autotune = _autotune.AutoTuneConfig()
        self._tuner = _autotune.AutoTuner(autotune) if autotune else None
        self._engine = UpdateEngine(
            precision=self.precision,
            backend=backend,
            numa_aware=numa_aware,
            num_threads=num_threads,
            chunk_size=chunk_size,
            stochastic_rounding=stochastic_rounding,
            seed=seed,
            overlap_writeback=overlap_writeback,
            timing=self._tuner is not None,
            algorithm=self._algorithm,
            transfer_buffer_bytes=None if transfer_buffer_mb is None else transfer_buffer_mb << 20,
            master_precision=self.master_precision,
        )
        self._engine.delay = delay_updates and delay_warmup_steps == 0
        self._finalizer = weakref.finalize(self, UpdateEngine.close, self._engine)
        self._small_groups = self._build_small_groups()
        for slot in self._ordered:
            self._note_transfer(slot)

    def _note_transfer(self, slot: ParamSlot) -> None:
        p = slot.param
        if p.is_cuda and p.requires_grad and slot.small is None:
            self._engine.note_transfer(p)

    # ---------------------------------------------------------------- groups

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group["params"]:
            if p.is_sparse:
                raise ValueError("AdamBridge does not support sparse parameters")
            slot = ParamSlot(len(self._ordered), p, group)
            self._slots[p] = slot
            self._ordered.append(slot)
            if self._hooked and p.requires_grad:
                self._hooks.append(p.register_post_accumulate_grad_hook(self._hook()))
            if hasattr(self, "_engine"):  # groups added after construction
                self._note_transfer(slot)

    def _build_small_groups(self) -> list[SmallGroup]:
        """Group small CUDA parameters by device and dtype (before any state exists)."""
        by_key: dict[tuple, list[ParamSlot]] = {}
        for slot in self._ordered:
            p = slot.param
            if p.is_cuda and p.requires_grad and slot.numel < self.small_param_threshold:
                by_key.setdefault((p.device, p.dtype), []).append(slot)
        return [SmallGroup(slots) for slots in by_key.values()]

    def _rebind_groups(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                self._slots[p].group = group

    # ------------------------------------------------------------ async mode

    def attach_hooks(self, model: torch.nn.Module | None = None) -> "AdamBridge":
        """Start updating each parameter as soon as its gradient is ready.

        If ``model`` is given, only its parameters are hooked.
        """
        if self._hooked:
            return self
        wanted = None if model is None else {id(p) for p in model.parameters()}
        hook = self._hook()
        for slot in self._ordered:
            p = slot.param
            if p.requires_grad and slot.small is None and (wanted is None or id(p) in wanted):
                self._hooks.append(p.register_post_accumulate_grad_hook(hook))
        self._hooked = True
        return self

    def _hook(self):
        """The gradient hook, opaque to TorchDynamo: compiled autograd calls it
        instead of tracing the copies, events and thread hand-offs inside."""
        return torch.compiler.disable(self._on_grad_ready)

    def detach_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._hooked = False

    @contextlib.contextmanager
    def no_sync(self):
        """Accumulate gradients without triggering updates (all but the last micro-batch)."""
        self._accumulating += 1
        try:
            yield
        finally:
            self._accumulating -= 1

    def _on_grad_ready(self, p: torch.Tensor) -> None:
        if self._accumulating or p.grad is None:
            return
        if self._slots[p].gpu is not None:
            return  # state on the GPU: updated at step_sync
        if self.accumulation_steps > 1:
            if not self._backward_counted:
                self._backward_counted = True
                torch.autograd.Variable._execution_engine.queue_callback(self._end_of_backward)
            if self._micro + 1 < self.accumulation_steps:
                return  # not the last micro-batch: let the gradient accumulate
        self._submit(self._slots[p], p.grad)
        if self.release_grads:
            p.grad = None

    def _end_of_backward(self) -> None:
        self._micro += 1
        self._backward_counted = False

    def _submit(self, slot: ParamSlot, grad: torch.Tensor) -> None:
        if slot.submitted:
            raise RuntimeError(
                "AdamBridge: parameter received a second gradient before step_sync(); "
                "run extra backward passes inside optimizer.no_sync() to accumulate gradients"
            )
        if grad.is_sparse:
            raise RuntimeError("AdamBridge does not support sparse gradients")
        slot.submitted = True
        if self.gpu_state_budget and not self._gpu_placed:
            self._grad_order.append(slot)
        self._engine.submit(slot, grad)

    def _flush_small(self, release: bool) -> None:
        """Queue the small parameters' updates, one transfer per group."""
        for group in self._small_groups:
            grads = [None if s.submitted else s.param.grad for s in group.slots]
            for slot in self._engine.submit_small(group, grads):
                slot.submitted = True
                if release:
                    slot.param.grad = None

    def step_sync(self) -> StepStats:
        """Update the small parameters, wait for all queued host updates, and
        order the weight write-backs before subsequent work on the current
        CUDA stream. With delayed updates, queue the updates and return."""
        try:
            if not self._accumulating:
                if self.accumulation_steps > 1:
                    self._submit_remaining()  # a window cut short: hooks have not fired
                self._flush_small(release=self.release_grads and self._hooked)
                self._step_gpu()  # queued on the GPU before waiting for the host: the two overlap
            stats = self._engine.finish_step()
        finally:
            for slot in self._ordered:
                slot.submitted = False
            self._micro = 0
        self.last_step_stats = stats
        self._steps_synced += 1
        if self.delay_updates and self._steps_synced >= self.delay_warmup_steps:
            self._engine.delay = True
        if self._tuner is not None and stats.num_updates:
            self._run_autotune(stats)
        if self.gpu_state_budget and not self._gpu_placed and not self._accumulating:
            self._place_on_gpu()
        return stats

    def _submit_remaining(self) -> None:
        for slot in self._ordered:
            if not slot.submitted and slot.small is None and slot.gpu is None and slot.param.grad is not None:
                self._submit(slot, slot.param.grad)

    # ----------------------------------------------------- state on the GPU

    def _gpu_budget(self, device: torch.device) -> int:
        if self.gpu_state_budget != "auto":
            return int(self.gpu_state_budget)
        free, total = torch.cuda.mem_get_info(device)
        # What the next step needs beyond what is reserved now (activations, gradients): the first step's peak.
        again = max(0, torch.cuda.max_memory_reserved(device) - torch.cuda.memory_reserved(device))
        return max(0, free - again - max(1 << 30, total // 20))

    def _place_on_gpu(self) -> None:
        """After the first step: move the state of the last parameters to get their gradients to the GPU."""
        self._gpu_placed = True
        self._engine.flush()
        order = list(dict.fromkeys(self._grad_order))
        self._grad_order = []
        budgets: dict[torch.device, int] = {}
        chosen: list[ParamSlot] = []
        for slot in reversed(order):  # the last gradients of backward first
            p = slot.param
            if not p.is_cuda or slot.small is not None or not slot.initialized:
                continue
            if p.device not in budgets:
                budgets[p.device] = self._gpu_budget(p.device)
            need = slot.numel * 18  # FP32 master, m, v; the gradient held until step_sync, in FP32 for the update
            if need > budgets[p.device]:
                continue
            budgets[p.device] -= need
            chosen.append(slot)
        by_group: dict[int, list[ParamSlot]] = {}
        for slot in chosen:
            device, shape = slot.param.device, slot.param.shape
            master = _master_fp32(slot).to(device).view(shape)
            m, v = slot.m.to_fp32().to(device).view(shape), slot.v.to_fp32().to(device).view(shape)
            self._engine.free_buffers(slot)
            slot.gpu, slot.m, slot.v = master, m, v  # FP32 tensors on the GPU while slot.gpu is set
            by_group.setdefault(id(slot.group), []).append(slot)
        for slots in by_group.values():
            self._add_gpu_group(slots)
        if chosen:
            log.info("adambridge: optimizer state of %d parameters (%.1fM elements) kept on the GPU",
                     len(chosen), sum(s.numel for s in chosen) / 1e6)

    def _add_gpu_group(self, slots: list[ParamSlot]) -> None:
        group = slots[0].group
        inner = torch.optim.AdamW([s.gpu for s in slots], lr=group["lr"], betas=group["betas"], eps=group["eps"],
                                  weight_decay=group["weight_decay"], fused=True)
        for s in slots:
            inner.state[s.gpu] = {"step": torch.tensor(float(s.step), dtype=torch.float32, device=s.gpu.device),
                                  "exp_avg": s.m, "exp_avg_sq": s.v}
        self._gpu_opts.append((group, inner, slots))

    @torch.no_grad()
    def _step_gpu(self) -> None:
        for group, inner, slots in self._gpu_opts:
            live = [s for s in slots if s.param.grad is not None]
            if not live:
                continue
            g = inner.param_groups[0]
            for k in ("lr", "betas", "eps", "weight_decay"):
                g[k] = group[k]
            for s in slots:
                s.gpu.grad = None if s.param.grad is None else s.param.grad.to(torch.float32)
            inner.step()
            torch._foreach_copy_([s.param.data for s in live], [s.gpu for s in live])  # round to nearest even
            for s in live:
                s.step += 1
                s.gpu.grad = None
                s.param.grad = None

    def flush(self) -> None:
        """Apply every pending update to the parameters and wait for the host.
        With ``delay_updates``, call it before evaluating or saving the model;
        otherwise it is a no-op beyond waiting."""
        self._engine.flush()

    # ------------------------------------------------------------- sync mode

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        """Update every parameter that has a gradient and was not already
        updated by a hook this iteration, then :meth:`step_sync`."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._submit_remaining()
        self._flush_small(release=False)
        self.step_sync()
        return loss

    # ------------------------------------------------------------- precision

    def set_precision(self, precision: str) -> None:
        """Re-encode all moments in another precision (between steps)."""
        precision = canonical_precision(precision)
        self._engine.wait()
        if precision == self.precision:
            return
        for slot in self._ordered:
            if slot.initialized and slot.gpu is None:  # state on the GPU stays FP32
                self._engine.convert_moments(slot, precision)
        self.precision = self._engine.precision = precision

    def _run_autotune(self, stats: StepStats) -> None:
        tuner = self._tuner
        if tuner.config.tune_threads and not tuner.threads_tuned:
            tuner.threads_tuned = True
            for device in {slot.param.device for slot in self._ordered}:
                _autotune.benchmark_threads(self._engine, device)
            return  # this step's timing included the benchmark's page faults
        new = tuner.observe(stats, self.precision)
        if new is not None:
            self.set_precision(new)

    @property
    def autotune_log(self) -> list[str]:
        return [] if self._tuner is None else list(self._tuner.decisions)

    def tune_threads(self, numel: int = 1 << 22) -> dict[str, dict[int, float]]:
        """Benchmark the update kernel and pick the fastest thread count per device."""
        return {
            str(device): _autotune.benchmark_threads(self._engine, device, numel)
            for device in {slot.param.device for slot in self._ordered}
        }

    # ------------------------------------------------------------ inspection

    @property
    def backend(self) -> str:
        """``'native/<isa>'`` (every moment precision, FP8 included) or ``'torch'``."""
        k = self._engine.native
        return f"native/{k.simd_level}" if k is not None else "torch"

    def memory_report(self) -> dict[str, int]:
        """Host bytes held by the optimizer, by category."""
        report = {"master": 0, "moments": 0, "staging": self._engine.transfer_bytes()}
        for group in self._small_groups:
            report["staging"] += sum(buf.nbytes for buf in group._grad_host.values())
        gpu = sum(s.numel * 12 for s in self._ordered if s.gpu is not None)
        for slot in self._ordered:
            if not slot.initialized or slot.gpu is not None:
                continue
            if not _master_in_place(slot):
                report["master"] += slot.master.nbytes
            report["moments"] += slot.m.nbytes + (slot.v.nbytes if slot.v is not None else 0)
            for buf in (slot.out, slot.grad_host):
                if buf is not None and buf.data_ptr() != slot.param.data_ptr():
                    report["staging"] += buf.nbytes
        report["total"] = sum(report.values())
        report["gpu_state"] = gpu  # on the GPU, not in the (host) total
        return report

    def expected_bytes_per_param(self) -> float:
        return bytes_per_param(self.precision, 1 if self._algorithm == "lion" else 2, master=self.master_precision)

    def master_params(self) -> list[torch.Tensor | None]:
        """FP32 master copies (``None`` until a parameter's first update; a decoded copy for an FP24 master)."""
        return [slot.gpu if slot.gpu is not None else _master_fp32(slot) if slot.initialized else None
                for slot in self._ordered]

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict[str, Any]:
        """Also applies pending delayed updates, so the model's weights match
        the saved master weights."""
        self._engine.flush()
        sd = super().state_dict()
        sd["state"] = {}
        for slot in self._ordered:
            if slot.gpu is not None:  # saved like host state, with FP32 moments (loading converts them)
                sd["state"][slot.index] = {
                    "step": slot.step, "master": slot.gpu.detach().reshape(-1).cpu(),
                    "exp_avg": make_store("fp32", slot.numel, "m", init=slot.m.reshape(-1).cpu()).state_dict(),
                    "exp_avg_sq": make_store("fp32", slot.numel, "v", init=slot.v.reshape(-1).cpu()).state_dict()}
            elif slot.initialized:
                master = slot.master.data if isinstance(slot.master, FP32Store) else slot.master.state_dict()
                st = sd["state"][slot.index] = {"step": slot.step, "master": master, "exp_avg": slot.m.state_dict()}
                if slot.v is not None:
                    st["exp_avg_sq"] = slot.v.state_dict()
        sd["adambridge"] = {"precision": self.precision, "master_precision": self.master_precision, "format": 1,
                            "algorithm": self._algorithm}
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load an AdamBridge/LionBridge checkpoint, or a ``torch.optim.AdamW`` one
        (moments are converted to this optimizer's precision)."""
        self._engine.discard_pending()
        saved = {int(k) if isinstance(k, str) and k.isdigit() else k: v for k, v in state_dict["state"].items()}
        lion = self._algorithm == "lion"
        # Validate everything before changing anything: a bad checkpoint must not leave a half-loaded optimizer.
        for index, st in saved.items():
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self._ordered):
                raise ValueError(f"checkpoint state for parameter {index!r}: this optimizer has parameters "
                                 f"0..{len(self._ordered) - 1}")
            numel = self._ordered[index].numel
            if st.get("master") is not None:
                check_master_state(st["master"], numel, f"master weights of parameter {index}")
            for key in ("exp_avg", "exp_avg_sq"):
                if isinstance(st.get(key), torch.Tensor):
                    _checked(st[key], numel, f"{key} of parameter {index}")
        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
        self._rebind_groups()  # Optimizer.load_state_dict replaces the group dicts
        for index, st in saved.items():
            slot = self._ordered[index]
            if not lion and "exp_avg_sq" not in st:
                raise ValueError("checkpoint has no exp_avg_sq (saved by LionBridge?); AdamW needs both moments")
            if slot.gpu is not None:
                self._load_gpu_slot(slot, st)
                continue
            if not slot.initialized:
                self._engine.initialize(slot)
            slot.step = int(st["step"])
            if st.get("master") is not None:
                load_master_(slot.master, st["master"])
            self._engine.replace_moments(
                slot, self._load_store(st["exp_avg"], slot.numel, "m"),
                None if lion else self._load_store(st["exp_avg_sq"], slot.numel, "v"))

    def _load_gpu_slot(self, slot: ParamSlot, st: dict) -> None:
        def fp32(saved, role):
            t = saved if isinstance(saved, torch.Tensor) else store_from_state_dict(saved, slot.numel, role).to_fp32()
            return t.reshape(slot.param.shape).to(slot.gpu.device, torch.float32)

        slot.step = int(st["step"])
        with torch.no_grad():
            if st.get("master") is not None:
                slot.gpu.copy_(master_to_fp32(st["master"], slot.numel).reshape(slot.param.shape))
            slot.m.copy_(fp32(st["exp_avg"], "m"))
            slot.v.copy_(fp32(st["exp_avg_sq"], "v"))
        for _, inner, slots in self._gpu_opts:
            if slot in slots:
                inner.state[slot.gpu]["step"].fill_(float(slot.step))

    def _load_store(self, saved, numel: int, role: str):
        if isinstance(saved, torch.Tensor):  # torch.optim.AdamW layout
            return make_store(self.precision, numel, role, init=saved.float())
        store = store_from_state_dict(saved, numel, role)
        if store.precision != self.precision:
            store = make_store(self.precision, numel, role, init=store.to_fp32())
        return store

    def close(self) -> None:
        """Apply pending delayed updates and stop worker threads. Called
        automatically (without applying) when the optimizer is collected."""
        self.detach_hooks()
        if self._finalizer.alive:
            try:
                self._engine.flush()
            finally:
                self._finalizer()


def _master_in_place(slot: ParamSlot) -> bool:
    """The slot's master is the (FP32, CPU) parameter itself."""
    return isinstance(slot.master, FP32Store) and slot.master.data.data_ptr() == slot.param.data_ptr()


def _master_fp32(slot: ParamSlot) -> torch.Tensor:
    """The slot's host master as a flat FP32 tensor: the live tensor for FP32, a decoded copy for FP24."""
    return slot.master.data if isinstance(slot.master, FP32Store) else slot.master.to_fp32()


OptiComputeAdamW = AdamBridge


class LionBridge(AdamBridge):
    r"""Lion (Chen et al., 2023, "Symbolic Discovery of Optimization Algorithms")
    with the same host offload, kernels and pipeline as :class:`AdamBridge`::

        c = beta1 * m + (1 - beta1) * g
        p = p * (1 - lr * weight_decay) - lr * sign(c)
        m = beta2 * m + (1 - beta2) * g

    Lion keeps one moment instead of two, so host state is 8 bytes per
    parameter with FP32 moments (AdamW: 12), 6 with BF16, about 5 with FP8,
    and each update moves less memory. Lion needs a smaller learning rate than
    AdamW (typically 3-10x) and a correspondingly larger weight decay; its
    convergence relative to AdamW depends on the task.

    Every keyword argument of :class:`AdamBridge` except ``eps`` applies
    (``precision``, ``delay_updates``, ``accumulation_steps``, ...). The
    moment is saved as ``exp_avg``.
    """

    _algorithm = "lion"

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        **kwargs,
    ):
        if "eps" in kwargs:
            raise TypeError("LionBridge has no eps")
        super().__init__(params, lr=lr, betas=betas, eps=0.0, weight_decay=weight_decay, **kwargs)
