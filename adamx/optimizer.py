"""AdamX: AdamW with optimizer state offloaded to host memory and updated by
SIMD kernels on the CPU, overlapped with the GPU backward pass."""

from __future__ import annotations

import contextlib
import logging
import weakref
from typing import Any, Callable, Iterable

import torch

from . import autotune as _autotune
from .engine import ParamSlot, SmallGroup, StepStats, UpdateEngine
from .precision import bytes_per_param, canonical_precision, make_store, store_from_state_dict

__all__ = ["AdamX", "OptiComputeAdamW"]

log = logging.getLogger("adamx")


class AdamX(torch.optim.Optimizer):
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
    accumulation, run all but the last micro-batch inside :meth:`no_sync`.

    Args:
        params: iterable of parameters or parameter groups.
        lr, betas, eps, weight_decay: as in ``torch.optim.AdamW``.
        precision: moment storage: ``'fp32'``, ``'bf16'`` (alias
            ``'bf16_moments'``) or ``'fp8'`` (alias ``'fp8_moments'``; E4M3 for
            ``m``, E5M2 for ``sqrt(v)``, per-128-element scales).
        stochastic_rounding: round low-precision moments stochastically.
            Disabling it makes BF16/FP8 moments stall for ``beta2`` near 1.
        backend: ``'auto'`` (native SIMD kernel when it can be built, else
            PyTorch), ``'native'`` or ``'torch'``.
        numa_aware: allocate state and pin update threads on the NUMA node
            each GPU is attached to (Linux).
        num_threads: update threads per NUMA node (default: its CPUs minus
            two, at most 32).
        chunk_size: elements per PyTorch-kernel chunk (default from L2 size).
        release_grads: in asynchronous mode, free each gradient once it has
            been copied to the host.
        overlap_writeback: copy updated weights to the GPU as soon as each is
            ready. If False, all copies are issued in :meth:`step_sync`. Turn
            it off if some op reads a parameter during backward *after* that
            parameter's gradient is complete (only possible via ``.detach()``
            tricks).
        autotune: ``True`` or an :class:`~adamx.autotune.AutoTuneConfig` to
            tune thread count and moment precision at runtime.
        seed: base seed for stochastic rounding.
        small_param_threshold: CUDA parameters with fewer elements (norms,
            biases) get no per-tensor hook; they are updated together at
            :meth:`step_sync` with one transfer each way. Their update takes
            microseconds, so overlapping it with backward gains nothing, while a
            hook per tensor costs tens of microseconds of Python each.
    """

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
    ):
        if not 0.0 <= lr:
            raise ValueError(f"invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"invalid weight_decay: {weight_decay}")
        if chunk_size is not None and (chunk_size <= 0 or chunk_size % 128):
            raise ValueError("chunk_size must be a positive multiple of 128")

        self.precision = canonical_precision(precision)
        self.release_grads = release_grads
        self.small_param_threshold = small_param_threshold
        self._slots: dict[torch.Tensor, ParamSlot] = {}
        self._ordered: list[ParamSlot] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._hooked = False
        self._accumulating = 0
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
        )
        self._finalizer = weakref.finalize(self, UpdateEngine.close, self._engine)
        self._small_groups = self._build_small_groups()

    # ---------------------------------------------------------------- groups

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for p in group["params"]:
            if p.is_sparse:
                raise ValueError("AdamX does not support sparse parameters")
            slot = ParamSlot(len(self._ordered), p, group)
            self._slots[p] = slot
            self._ordered.append(slot)
            if self._hooked and p.requires_grad:
                self._hooks.append(p.register_post_accumulate_grad_hook(self._on_grad_ready))

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

    def attach_hooks(self, model: torch.nn.Module | None = None) -> "AdamX":
        """Start updating each parameter as soon as its gradient is ready.

        If ``model`` is given, only its parameters are hooked.
        """
        if self._hooked:
            return self
        wanted = None if model is None else {id(p) for p in model.parameters()}
        for slot in self._ordered:
            p = slot.param
            if p.requires_grad and slot.small is None and (wanted is None or id(p) in wanted):
                self._hooks.append(p.register_post_accumulate_grad_hook(self._on_grad_ready))
        self._hooked = True
        return self

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
        self._submit(self._slots[p], p.grad)
        if self.release_grads:
            p.grad = None

    def _submit(self, slot: ParamSlot, grad: torch.Tensor) -> None:
        if slot.submitted:
            raise RuntimeError(
                "AdamX: parameter received a second gradient before step_sync(); "
                "run extra backward passes inside optimizer.no_sync() to accumulate gradients"
            )
        if grad.is_sparse:
            raise RuntimeError("AdamX does not support sparse gradients")
        slot.submitted = True
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
        CUDA stream."""
        try:
            if not self._accumulating:
                self._flush_small(release=self.release_grads and self._hooked)
            stats = self._engine.finish_step()
        finally:
            for slot in self._ordered:
                slot.submitted = False
        self.last_step_stats = stats
        if self._tuner is not None and stats.num_updates:
            self._run_autotune(stats)
        return stats

    # ------------------------------------------------------------- sync mode

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        """Update every parameter that has a gradient and was not already
        updated by a hook this iteration, then :meth:`step_sync`."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for slot in self._ordered:
            if not slot.submitted and slot.small is None and slot.param.grad is not None:
                self._submit(slot, slot.param.grad)
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
            if slot.initialized:
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
        report = {"master": 0, "moments": 0, "staging": 0}
        for group in self._small_groups:
            report["staging"] += sum(buf.nbytes for buf in group._grad_host.values())
        for slot in self._ordered:
            if not slot.initialized:
                continue
            if slot.master.data_ptr() != slot.param.data_ptr():
                report["master"] += slot.master.nbytes
            report["moments"] += slot.m.nbytes + slot.v.nbytes
            for buf in (slot.out, slot.grad_host):
                if buf is not None and buf.data_ptr() != slot.param.data_ptr():
                    report["staging"] += buf.nbytes
        report["total"] = sum(report.values())
        return report

    def expected_bytes_per_param(self) -> float:
        return bytes_per_param(self.precision)

    def master_params(self) -> list[torch.Tensor | None]:
        """FP32 master copies (``None`` until a parameter's first update)."""
        return [slot.master for slot in self._ordered]

    # ------------------------------------------------------------ checkpoint

    def state_dict(self) -> dict[str, Any]:
        self._engine.wait()
        sd = super().state_dict()
        sd["state"] = {
            slot.index: {
                "step": slot.step,
                "master": slot.master,
                "exp_avg": slot.m.state_dict(),
                "exp_avg_sq": slot.v.state_dict(),
            }
            for slot in self._ordered
            if slot.initialized
        }
        sd["adamx"] = {"precision": self.precision, "format": 1}
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load an AdamX checkpoint, or a ``torch.optim.AdamW`` one (moments are
        converted to this optimizer's precision)."""
        self._engine.wait()
        saved = state_dict["state"]
        super().load_state_dict({"state": {}, "param_groups": state_dict["param_groups"]})
        self._rebind_groups()  # Optimizer.load_state_dict replaces the group dicts
        for index, st in saved.items():
            slot = self._ordered[int(index)]
            if not slot.initialized:
                self._engine.initialize(slot)
            slot.step = int(st["step"])
            if st.get("master") is not None:
                slot.master.copy_(st["master"].reshape(-1))
            self._engine.replace_moments(slot, self._load_store(st["exp_avg"], slot.numel, "m"),
                                         self._load_store(st["exp_avg_sq"], slot.numel, "v"))

    def _load_store(self, saved, numel: int, role: str):
        if isinstance(saved, torch.Tensor):  # torch.optim.AdamW layout
            return make_store(self.precision, numel, role, init=saved.float())
        store = store_from_state_dict(saved, numel, role)
        if store.precision != self.precision:
            store = make_store(self.precision, numel, role, init=store.to_fp32())
        return store

    def close(self) -> None:
        """Stop worker threads. Called automatically when the optimizer is collected."""
        self.detach_hooks()
        self._finalizer()


OptiComputeAdamW = AdamX
