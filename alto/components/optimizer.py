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

LPT modules expose their plain parameters through
:meth:`~alto.nn.lpt.LPTModule.iter_lpt_weights`.  The hook uses that
``TrainingOpConfig`` and the module's declared QDQ axis.  Embeddings,
norms, biases, and MXFP8 weights are skipped.

"""


from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torchtitan.components.optim import OptimizersContainer

from alto.config import TrainingOpConfig
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "DeOscillationConfig",
    "enable_de_oscillation",
]


QdqFn = Callable[[torch.Tensor, int], torch.Tensor]


def _make_qdq_fn_for(cfg: TrainingOpConfig) -> QdqFn:
    """Build a ``dequant(quant(., axis))`` callable matching the forward QDQ.

    The closure runs on the plain local FP32/BF16 storage.  The reduction
    axis is passed in by the caller from the owning LPT module.
    """
    from alto.kernels.fp4 import (
        convert_from_amdfp4,
        convert_from_mxfp4,
        convert_from_nvfp4,
        convert_to_amdfp4,
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
                w, axis=axis, is_2d_block=is_2d_block, use_uos=cfg.use_uos
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

    elif cfg.precision in ("nvfp4", "amdfp4"):
        use_outer_scale = cfg.two_level_scaling == "tensorwise"
        # AMD-FP4 shares the NVFP4 micro-block layout (16-element blocks +
        # FP32 outer scale); the only difference is the inner-scale grid,
        # which is pinned by *which* convert ops we call (amdfp4 → UE5M3,
        # nvfp4 → E4M3).  Everything else (padding, outer scale, axis) is
        # identical, so the closure below is shared.
        if cfg.precision == "amdfp4":
            convert_to_fp4, convert_from_fp4 = convert_to_amdfp4, convert_from_amdfp4
        else:
            convert_to_fp4, convert_from_fp4 = convert_to_nvfp4, convert_from_nvfp4

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

            data_lp, scales = convert_to_fp4(
                w, axis=axis, is_2d_block=is_2d_block,
                outer_scale=outer_scale,
                update_outer_scale=use_outer_scale,
            )
            dequantized = convert_from_fp4(
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
            f"de-oscillation only supports FP4 precisions, "
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

    def __init__(
        self,
        cfg: DeOscillationConfig,
        model_parts: list[nn.Module] | None = None,
    ) -> None:
        self._cfg = cfg
        # TrainingOpConfig is @dataclass(unsafe_hash=True), so the same
        # config object (or two equal ones) maps to a single cached QDQ.
        # The cached closure takes ``(tensor, axis)``.  The axis comes from
        # the owning LPT module.
        self._qdq_cache: dict[TrainingOpConfig, QdqFn] = {}
        self._lpt_weights = _index_lpt_weights(model_parts or [])
        # Counts how many periods have completed since the hook was
        # installed; used only for log throttling.
        self._periods_completed = 0

    # ---- per-config QDQ ----------------------------------------------

    def _qdq_for(self, cfg: TrainingOpConfig) -> QdqFn:
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
                tracked = self._tracked_weight(p)
                if tracked is None:
                    continue
                raw, op_config, axis = tracked
                state = optimizer.state[p]
                reset_count, total_count, did_reset = self._step_storage(
                    raw, op_config, axis, state
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
                logger.info(f'[de-osc] period#{self._periods_completed} triggered_params={triggered_params} reset_ratio={pct:.4f}% ({n_reset}/{n_total})')

    # ---- per-parameter update ----------------------------------------

    def _tracked_weight(
        self,
        param: nn.Parameter,
    ) -> tuple[torch.Tensor, TrainingOpConfig, int] | None:
        """Return local storage, config, and QDQ axis for an eligible parameter."""
        if not param.requires_grad:
            return None
        tracked = self._lpt_weights.get(id(param))
        if tracked is None:
            return None
        op_config, axis = tracked
        return _local_storage(param), op_config, axis

    def _step_storage(
        self,
        raw: torch.Tensor,
        op_config: TrainingOpConfig,
        axis: int,
        state: dict[str, Any],
    ) -> tuple[int, int, bool]:
        """Track movement, maybe reset, return ``(n_reset, n_total, did_reset)``."""
        cfg = self._cfg
        # ``raw`` is the plain local FP32/BF16 storage.
        w_now = raw.detach().bfloat16()

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

        qdq_fn = self._qdq_for(op_config)
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
            # Snap into the parameter storage in place; the
            # FP4 GEMM dispatch on the next forward will read this new
            # value.
            raw[reset_mask] = w_qdq[reset_mask].float()
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
        device with the parameter storage.  They are stored
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


def _local_storage(param: nn.Parameter) -> torch.Tensor:
    """Return the tensor storage owned by this rank."""
    data = param.data
    if isinstance(data, DTensor):
        return data._local_tensor
    return data


def _index_lpt_weights(
    model_parts: list[nn.Module],
) -> dict[int, tuple[TrainingOpConfig, int]]:
    from alto.nn.lpt import LPTModule

    specs: dict[int, tuple[TrainingOpConfig, int]] = {}
    for part in model_parts:
        for module in part.modules():
            if not isinstance(module, LPTModule):
                continue
            for param, op_config, axis in module.iter_lpt_weights():
                if op_config.precision not in ("mxfp4", "nvfp4", "amdfp4"):
                    continue
                specs[id(param)] = (op_config, axis)
    return specs


def enable_de_oscillation(
    optimizers: OptimizersContainer,
    config: DeOscillationConfig,
    model_parts: list[nn.Module] | None = None,
) -> None:
    if not config.enable or de_oscillation_enabled(optimizers):
        return
    hook = _DeOscillationHook(config, model_parts)
    for opt in optimizers.optimizers:
        opt.register_step_post_hook(hook)
    logger.info(f'[de-osc] enabled period={config.period} ratio_threshold={config.ratio_threshold} log_freq={config.log_freq}')


def de_oscillation_enabled(optimizers: OptimizersContainer):
    """
    Check if weight de-oscillation is enabled for any optimizer in the container.
    """
    for opt in optimizers.optimizers:
        hooks = getattr(opt, "_optimizer_step_post_hooks", {})
        if any(isinstance(h, _DeOscillationHook) for h in hooks.values()):
            return True
    return False
