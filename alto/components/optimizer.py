# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Weight de-oscillation hook for FP4-aware AdamW.

This module provides :class:`DeOscillationConfig` and :func:`enable_de_oscillation`,
which install an optimizer ``step_post_hook`` implementing the weight de-oscillation strategy.

Motivation
----------

When training with simulated FP4 GEMMs, the FP32 master weight is
re-quantized to FP4 just before every matmul.  An element whose master
value sits near a quantization-bin boundary can *oscillate*: tiny FP
gradient steps flip the bin assignment on every iteration, so although
the master weight moves only slightly the *quantized* weight seen by the
GEMM hops between two adjacent bins every step.  This hurts convergence.

Detection criterion (DistRatio)
-------------------------------

For each weight element, over a window of ``period`` optimizer steps,
accumulate two L1 distances of consecutive iterations:

* ``dist_w     = sum_t |w_t       - w_{t-1}|``     (raw FP movement)
* ``dist_w_qdq = sum_t |Q(w_t)    - Q(w_{t-1})|``  (quantized movement)

where ``Q`` is the project's FP4 QDQ round trip.  An element is treated
as oscillating when ``dist_w_qdq / dist_w >= ratio_threshold``: small
FP movement producing large quantized movement only happens at a bin
boundary.

Reset
-----

On the last step of each period, oscillating elements are *snapped* to
their current bin center, i.e. ``w <- Q(w)``.  Counters and snapshots
are then re-initialized for the next period.

Scope: which parameters are eligible
------------------------------------

The hook only acts on parameters whose storage is wrapped by either
:class:`~alto.kernels.dispatch.tensor.MXFP4TrainingWeightWrapperTensor` or
:class:`~alto.kernels.dispatch.tensor.NVFP4TrainingWeightWrapperTensor`
(directly, or as ``DTensor._local_tensor`` under FSDP2).  Other
parameters (embeddings, norm scales, biases, MXFP8 weights, ...) are
silently skipped.  The FP4 format and the per-block scaling layout used
for ``Q`` come from the wrapper's
:class:`~alto.kernels.dispatch.config.TrainingOpConfig`, and the
reduction axis along which QDQ is performed comes from
``_infer_reduction_axis``.  Reading both from the wrapper guarantees that the
de-oscillation hook tracks oscillation in exactly the same FP4 grid the
forward / backward GEMMs see, with no axis inference duplicated here.

"""


from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.logging import logger

from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.tensor import (
    MXFP4TrainingWeightWrapperTensor,
    NVFP4TrainingWeightWrapperTensor,
    TrainingWeightWrapperBaseTensor,
)

__all__ = [
    "DeOscillationConfig",
    "enable_de_oscillation",
]


# Wrappers eligible for de-oscillation.  Only the two FP4 variants are
# included on purpose: MXFP8 has a much wider grid where oscillation is
# rare, and the bin-center snap we use only really makes sense for FP4.
_FP4_WRAPPER_TYPES: tuple[type, ...] = (
    MXFP4TrainingWeightWrapperTensor,
    NVFP4TrainingWeightWrapperTensor,
)


def _infer_reduction_axis(w: torch.Tensor):
    if w.dim() == 2:
        return -1
    elif w.dim() == 3:
        return -2
    else:
        raise ValueError(f"Unexpected weight shape: {w.shape}")


def _peel_to_fp4_wrapper(
    p: torch.Tensor,
) -> TrainingWeightWrapperBaseTensor | None:
    """Return the FP4 wrapper tensor backing *p*, or ``None``.

    Handles three layouts:

    * plain ``nn.Parameter(FP4WrapperTensor(...))``
    * FSDP2 ``nn.Parameter(DTensor(local=FP4WrapperTensor(...)))``
    * any of the above un-parameterised (e.g. inside checkpointing utils)
    """
    t = p.data if isinstance(p, nn.Parameter) else p
    if isinstance(t, DTensor):
        t = t._local_tensor
    if isinstance(t, _FP4_WRAPPER_TYPES):
        return t  # type: ignore[return-value]
    return None


QdqFn = Callable[[torch.Tensor, int], torch.Tensor]


def _make_qdq_fn_for(cfg: TrainingOpConfig) -> QdqFn:
    """Build a ``dequant(quant(., axis))`` callable matching the wrapper's
    forward-path QDQ.

    The closure runs on the *plain* underlying FP32/BF16 tensor (i.e.
    ``wrapper._data``), with the same ``is_2d_block`` choice the FP4
    GEMM applies to the weight operand.  The reduction axis is passed in
    by the caller (read from ``_infer_reduction_axis``) so the
    same cache entry serves both ``nn.Linear`` weights (axis ``-1``) and
    grouped-MM expert weights (axis ``-2``) that share a config.
    """
    from alto.kernels.fp4 import (
        convert_from_mxfp4,
        convert_from_nvfp4,
        convert_to_mxfp4,
        convert_to_nvfp4,
    )

    is_2d_block = cfg.use_2dblock_w

    if cfg.precision == "mxfp4":

        def qdq(w: torch.Tensor, axis: int) -> torch.Tensor:
            # hotfix for GPT-OSS 20B model
            original_rows = -1
            if w.dim() == 2 and w.shape[0] % 32 != 0:
                # for 2-D w, if the first dim is not divisible by 32, pad it to be divisible by 32
                original_rows = w.shape[0]
                w = torch.nn.functional.pad(w, (0, 0, 0, 32 - original_rows % 32))

            data_lp, scales = convert_to_mxfp4(
                w, axis=axis, is_2d_block=is_2d_block,
            )
            dequantized = convert_from_mxfp4(
                data_lp,
                scales,
                output_dtype=w.dtype,
                axis=axis,
                is_2d_block=is_2d_block,
            )
            # hotfix for GPT-OSS 20B model
            if original_rows != -1:
                dequantized = dequantized[:original_rows, :]
            return dequantized

    elif cfg.precision == "nvfp4":
        use_outer_scale = cfg.two_level_scaling == "tensorwise"

        def qdq(w: torch.Tensor, axis: int) -> torch.Tensor:
            # hotfix for GPT-OSS 20B model
            original_rows = -1
            if w.dim() == 2 and w.shape[0] % 16 != 0:
                # for 2-D w, if the first dim is not divisible by 16, pad it to be divisible by 16
                original_rows = w.shape[0]
                w = torch.nn.functional.pad(w, (0, 0, 0, 16 - original_rows % 16))

            outer_scale = (
                torch.empty(1, dtype=torch.float32, device=w.device)
                if use_outer_scale
                else None
            )

            data_lp, scales = convert_to_nvfp4(
                w, axis=axis, is_2d_block=is_2d_block,
                outer_scale=outer_scale,
                update_outer_scale=use_outer_scale,
            )
            dequantized = convert_from_nvfp4(
                data_lp,
                scales,
                output_dtype=w.dtype,
                axis=axis,
                is_2d_block=is_2d_block,
                outer_scale=outer_scale,
            )
            # hotfix for GPT-OSS 20B model
            if original_rows != -1:
                dequantized = dequantized[:original_rows, :]
            return dequantized
    else:
        raise ValueError(
            f"de-oscillation only supports FP4 wrappers, "
            f"got precision={cfg.precision!r}"
        )

    return qdq


@dataclass(kw_only=True, slots=True)
class DeOscillationConfig:
    """Per-period weight de-oscillation knobs.

    The FP4 format / block layout used for the QDQ round trip is *not*
    in this config -- it is read directly from each parameter's
    :class:`TrainingOpConfig` so the hook always operates on the same
    quantization grid that the forward pass uses.

    Attributes:
        enable: master switch.
        period: number of ``optimizer.step()`` calls per de-oscillation
            window.  Reset decisions are taken on the last step of each
            window.
        ratio_threshold: an element is reset when
            ``dist_w_qdq / dist_w >= ratio_threshold`` over the window.
            Reasonable values are in [4, 32]; the TetraJet reference
            uses ``16``.
        log_freq: log reset statistics every ``log_freq`` triggered
            periods.  ``0`` disables logging.
    """

    enable: bool = False
    period: int = 25
    ratio_threshold: float = 16.0
    log_freq: int = 0


class _DeOscillationHook:
    """AdamW ``step_post_hook`` implementing the DistRatio reset.

    The hook itself is nearly stateless (only an internal counter used
    for log throttling and a per-:class:`TrainingOpConfig` QDQ cache).
    All per-parameter state lives in ``optimizer.state[p]`` under keys
    prefixed with ``osci_``, so it is serialised through the standard
    optimizer state dict path.
    """

    KEY_STEP = "osci_step_in_period"
    KEY_DIST_FP = "osci_dist_w"
    KEY_DIST_QDQ = "osci_dist_w_qdq"
    KEY_PREV = "osci_prev_weight"

    # Floor for the per-element ratio denominator.  Picks a value well
    # above FP32 denormals so the divide stays in the normal range when
    # the corresponding element happened to be frozen for the whole
    # period.  The explicit ``dist_w > 0`` mask below makes this a pure
    # numeric safety net.
    _EPS = 1.0e-30

    def __init__(self, cfg: DeOscillationConfig) -> None:
        self._cfg = cfg
        # TrainingOpConfig is @dataclass(unsafe_hash=True), so the same
        # config object (or two equal ones) maps to a single cached QDQ.
        # The cached closure takes ``(tensor, axis)``; the axis is read
        # per-call from ``_infer_reduction_axis`` and is not
        # baked into the cache key.
        self._qdq_cache: dict[TrainingOpConfig, QdqFn] = {}
        # Counts how many periods have completed since the hook was
        # installed; used only for log throttling.
        self._periods_completed = 0

    # ---- eligibility --------------------------------------------------

    def _wrapper_if_eligible(
        self,
        p: nn.Parameter,
    ) -> TrainingWeightWrapperBaseTensor | None:
        """Return the FP4 wrapper if *p* is eligible for de-oscillation.

        Block-size divisibility is *not* re-checked here -- the FP4
        ``Linear`` / grouped-MM dispatch enforces it on every forward
        already, so any param that survives one training step is
        guaranteed to be compatible with the same QDQ kernels we are
        about to call.
        """
        if not p.requires_grad:
            return None
        wrapper = _peel_to_fp4_wrapper(p)
        if wrapper is None:
            return None
        cfg = wrapper.config
        if cfg is None or cfg.precision not in ("mxfp4", "nvfp4"):
            return None
        return wrapper

    # ---- per-config QDQ ----------------------------------------------

    def _qdq_for(self, wrapper: TrainingWeightWrapperBaseTensor) -> QdqFn:
        cfg = wrapper.config
        fn = self._qdq_cache.get(cfg)
        if fn is None:
            fn = _make_qdq_fn_for(cfg)
            self._qdq_cache[cfg] = fn
        return fn

    # ---- hook entry point --------------------------------------------

    def __call__(self, optimizer, args, kwargs) -> None:  # noqa: ARG002
        triggered_params = 0
        n_reset = 0
        n_total = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                wrapper = self._wrapper_if_eligible(p)
                if wrapper is None:
                    continue
                state = optimizer.state[p]
                reset_count, total_count, did_reset = self._step_param(
                    wrapper, state
                )
                if did_reset:
                    triggered_params += 1
                    n_reset += reset_count
                    n_total += total_count

        if triggered_params and self._cfg.log_freq:
            self._periods_completed += 1
            if (
                self._periods_completed % self._cfg.log_freq == 0
                and n_total > 0
            ):
                pct = n_reset / n_total * 100.0
                logger.info(
                    f"[de-osc] period#{self._periods_completed} "
                    f"triggered_params={triggered_params} "
                    f"reset_ratio={pct:.4f}% ({n_reset}/{n_total})"
                )

    # ---- per-parameter update ----------------------------------------

    def _step_param(
        self,
        wrapper: TrainingWeightWrapperBaseTensor,
        state: dict[str, Any],
    ) -> tuple[int, int, bool]:
        """Track movement, maybe reset, return ``(n_reset, n_total, did_reset)``."""
        cfg = self._cfg
        # ``raw`` is the plain FP32/BF16 storage backing the wrapper.
        # All math runs on it directly so we never accidentally go
        # through __torch_function__ (which would re-enter the FP4 GEMM
        # dispatch).
        raw = wrapper._data
        w_now = raw.detach()

        if self.KEY_PREV not in state or self.KEY_STEP not in state:
            # First call (either ever, or after a state-dict mishap).
            # Seed the period with the current weight and skip tracking
            # for this iteration; the next post_step will see a real
            # before/after pair.
            self._init_period_state(state, w_now)
            return 0, 0, False

        prev = state[self.KEY_PREV]
        dist_w = state[self.KEY_DIST_FP]
        dist_w_qdq = state[self.KEY_DIST_QDQ]

        qdq_fn = self._qdq_for(wrapper)
        axis = _infer_reduction_axis(w_now)
        w_qdq = qdq_fn(w_now, axis)
        prev_qdq = qdq_fn(prev, axis)
        dist_w.add_((w_now - prev).abs())
        dist_w_qdq.add_((w_qdq - prev_qdq).abs())
        prev.copy_(w_now)
        state[self.KEY_STEP] += 1

        if state[self.KEY_STEP] < cfg.period:
            return 0, 0, False

        # End-of-period decision: snap oscillating elements to their
        # current bin center, then start a fresh period.
        ratio = dist_w_qdq / dist_w.clamp(min=self._EPS)
        reset_mask = (dist_w > 0) & (ratio >= cfg.ratio_threshold)

        reset_count = 0
        total_count = dist_w.numel()
        if reset_mask.any():
            reset_count = int(reset_mask.sum().item())
            # Snap into the wrapper's underlying storage in place; the
            # FP4 GEMM dispatch on the next forward will read this new
            # value.
            raw[reset_mask] = w_qdq[reset_mask]
            # Refresh the snapshot so the next period does not see the
            # snap-to-bin-center as a large FP movement on its first
            # step.
            prev.copy_(raw)

        dist_w.zero_()
        dist_w_qdq.zero_()
        state[self.KEY_STEP] = 0
        return reset_count, total_count, True

    def _init_period_state(
        self,
        state: dict[str, Any],
        w: torch.Tensor,
    ) -> None:
        """Allocate (or zero) per-parameter de-oscillation state.

        State tensors are *plain* tensors that share dtype, shape and
        device with the wrapper's underlying storage.  They are stored
        in ``optimizer.state[p]`` directly so they ride along with the
        AdamW moments through the standard optimizer state dict path.
        """
        state[self.KEY_STEP] = 0
        if self.KEY_DIST_FP not in state:
            state[self.KEY_DIST_FP] = torch.zeros_like(w)
            state[self.KEY_DIST_QDQ] = torch.zeros_like(w)
            state[self.KEY_PREV] = w.clone()
        else:
            state[self.KEY_DIST_FP].zero_()
            state[self.KEY_DIST_QDQ].zero_()
            state[self.KEY_PREV].copy_(w)


def enable_de_oscillation(optimizers: OptimizersContainer, config: DeOscillationConfig) -> None:
    if not config.enable or de_oscillation_enabled(optimizers):
        return
    hook = _DeOscillationHook(config)
    for opt in optimizers.optimizers:
        opt.register_step_post_hook(hook)
    logger.info(
        "[de-osc] enabled "
        f"period={config.period} "
        f"ratio_threshold={config.ratio_threshold} "
        f"log_freq={config.log_freq}"
    )


def de_oscillation_enabled(optimizers: OptimizersContainer):
    """
    Check if weight de-oscillation is enabled for any optimizer in the container.
    """
    for opt in optimizers.optimizers:
        if any(isinstance(h, _DeOscillationHook) for h in opt._optimizer_step_post_hooks.values()):
            return True
    return False
