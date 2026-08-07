# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""AdaHOP calibration + per-slot Hadamard mode selection modifier.

Strategy-2 design (see /home/alirezak/han_branch/porting_notes.txt). A SINGLE
wrapper class (:class:`MXFP4AdaHOPWrapper`) is used from model-conversion time.
During calibration its modes are all ``"none"`` (== plain MXFP4); when
calibration finishes the modes are set IN PLACE (no wrapper-type swap, no new
``nn.Parameter``), so optimizer state references stay valid.

Lifecycle:

1. ``LowPrecisionTrainingModifier`` (scheme=``"mxfp4_adahop"``) wraps every
   targeted weight in :class:`MXFP4AdaHOPWrapper` (modes ``"none"``).
2. This modifier owns the Phase-A -> Phase-B transition:

   * ``on_convert`` configures the HadamardFactory.
   * ``on_initialize`` discovers the wrapped linears. (If ``transform_config_path``
     is set, pre-baked modes are applied here and calibration is skipped.)
   * ``on_pre_step`` — on the FIRST call (which happens AFTER
     ``checkpointer.load`` in the forge trainer), it inspects the checkpointed
     :class:`CalibrationStateManager`:
       - calibration already completed -> apply the restored modes in place and
         skip calibration (clean resume);
       - otherwise -> arm transient *module* observation hooks and begin
         calibration. It then opens a fresh per-step pattern bucket.
   * ``on_post_step`` aggregates after ``calibration_steps``, maps patterns to
     per-slot modes, applies them in place, records them in the
     CalibrationStateManager (so they ride the next checkpoint), writes the
     JSON artifact, and removes the observation hooks.

Calibration observation uses transient forward-pre / full-backward hooks on the
``nn.Linear`` modules — no closures are stored on tensors, so checkpoints remain
picklable.
"""

from typing import Any, Callable, Dict, List, Optional

import torch
from pydantic import Field, PrivateAttr
from torch import nn
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.modifiers.lpt.adahop_internals.calibration_hooks import (
    load_modes_from_json,
    make_backward_hook,
    make_forward_pre_hook,
    write_modes_json,
)
from alto.modifiers.lpt.adahop_internals.calibration_state import (
    get_calibration_modes,
    is_calibration_completed,
    set_calibration_result,
)

__all__ = ["AdaHOPModifier"]

_NONE_MODES = {"forward_y": "none", "backward_gx": "none", "backward_gw": "none"}


class AdaHOPModifier(Modifier):
    """Outlier-pattern-aware Hadamard calibration over MXFP4-wrapped linears."""

    enabled: bool = True

    use_hadamard: bool = True
    use_randomized_hadamard: bool = False

    calibration_steps: int = 30
    """Number of pre-step observations before transitioning to Phase B."""

    layer_transform_config: Dict[str, str] = Field(default_factory=dict)
    """Pattern-pair → ``TransformMode`` lookup. Keys are ``"row-row"``,
    ``"col-col"``, etc.; values are members of ``TransformMode``."""

    transform_config_path: Optional[str] = None
    """If set, load a pre-baked per-layer mode JSON and skip calibration.
    Format: ``{layer_fqn: {forward_y, backward_gx, backward_gw}}``."""

    dump_json_path: Optional[str] = None
    """If set, write the aggregated per-layer modes here after Phase B."""

    _step_idx: int = PrivateAttr(default=0)
    _per_step_patterns: List[Dict[str, Dict[str, str]]] = PrivateAttr(default_factory=list)
    _fqn_to_wrapper: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _fqn_to_module: Dict[str, nn.Module] = PrivateAttr(default_factory=dict)
    _handles: List[Any] = PrivateAttr(default_factory=list)
    _phase_b_done: bool = PrivateAttr(default=False)
    _resume_decided: bool = PrivateAttr(default=False)
    _calibration_armed: bool = PrivateAttr(default=False)

    @property
    def requires_training_mode(self) -> bool:
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        # The wrapper swap (to MXFP4AdaHOPWrapper, modes "none") is performed by
        # LowPrecisionTrainingModifier when scheme="mxfp4_adahop". Configure the
        # HadamardFactory here so the transform object is available later.
        if not self.enabled:
            logger.info("[AdaHOP] Modifier disabled (enabled=False); pass-through only.")
            return True
        logger.info(f"[AdaHOP] Modifier active: use_hadamard={self.use_hadamard}, "
                    f"use_randomized_hadamard={self.use_randomized_hadamard}, "
                    f"calibration_steps={self.calibration_steps}, "
                    f"transform_config_path={self.transform_config_path}, "
                    f"dump_json_path={self.dump_json_path or '<default ./outputs/adahop_calibration.json>'}, "
                    f"layer_transform_config entries={len(self.layer_transform_config)}")
        if not self.use_hadamard:
            logger.info("[AdaHOP] use_hadamard=False; skipping HadamardFactory configuration.")
            return True
        from alto._adahop_bridge import HadamardFactory
        HadamardFactory.configure(randomized=self.use_randomized_hadamard)
        logger.info(f"[AdaHOP] HadamardFactory configured (randomized={self.use_randomized_hadamard}).")
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.enabled:
            return True
        from alto.kernels.dispatch.adahop_tensor import MXFP4AdaHOPWrapper

        self._collect_wrappers(model_parts, MXFP4AdaHOPWrapper)
        n_wrappers = len(self._fqn_to_wrapper)
        logger.info(f"[AdaHOP] Discovered {n_wrappers} MXFP4AdaHOPWrapper-wrapped linears.")
        if n_wrappers > 0:
            sample = list(self._fqn_to_wrapper.keys())[:5]
            logger.info(f"[AdaHOP] First {len(sample)} FQNs: {', '.join(sample)}"
                        f"{' ...' if n_wrappers > 5 else ''}")
        elif not self._fqn_to_wrapper:
            logger.warning("[AdaHOP] No MXFP4AdaHOPWrapper found; "
                           "is LowPrecisionTrainingModifier(scheme='mxfp4_adahop') in the recipe?")

        # Manual pre-baked modes override: applied now (before checkpointer.load),
        # in place. Calibration is then skipped entirely.
        if self.transform_config_path is not None:
            modes_by_fqn = load_modes_from_json(self.transform_config_path)
            logger.info(f"[AdaHOP] Pre-baked modes: {len(modes_by_fqn)} loaded from "
                        f"{self.transform_config_path}; applying in place (calibration skipped).")
            self._log_mode_table(modes_by_fqn, aggregated=None)
            self._apply_modes_in_place(modes_by_fqn)
            self._resume_decided = True
        # Otherwise: defer the calibrate-vs-resume decision to the first
        # on_pre_step, which runs AFTER checkpointer.load() so the restored
        # CalibrationStateManager is visible.
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.enabled or self._phase_b_done:
            return True

        # First step after checkpointer.load(): decide resume vs. calibrate.
        if not self._resume_decided:
            self._resume_decided = True
            if is_calibration_completed():
                modes_by_fqn = get_calibration_modes()
                logger.info(f"[AdaHOP] Calibration restored from checkpoint "
                            f"({len(modes_by_fqn)} layers). Applying modes in place; "
                            f"skipping re-calibration.")
                self._log_mode_table(modes_by_fqn, aggregated=None)
                self._apply_modes_in_place(modes_by_fqn)
                return True
            # Fresh calibration.
            self._arm_calibration()

        if self._step_idx >= self.calibration_steps:
            return True
        # Open a fresh dict for this step's pattern observations.
        self._per_step_patterns.append({})
        logger.info(f"[AdaHOP] Calibration step {self._step_idx + 1}/{self.calibration_steps} opening bucket.")
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.enabled or self._phase_b_done:
            return True
        if not self._calibration_armed:
            return True
        self._step_idx += 1

        # Per-step observation summary
        total_layers = len(self._fqn_to_wrapper)
        bucket = self._per_step_patterns[-1] if self._per_step_patterns else {}
        layers_with_data = len(bucket)
        nx = sum(1 for v in bucket.values() if "x" in v)
        nw = sum(1 for v in bucket.values() if "w" in v)
        ng = sum(1 for v in bucket.values() if "grad_output" in v)
        logger.info(f"[AdaHOP] Calibration step {self._step_idx}/{self.calibration_steps} observed: "
                    f"{layers_with_data}/{total_layers} layers reported patterns. "
                    f"(x, w, grad_output) presence: {nx}/{nw}/{ng}.")

        if self._step_idx < self.calibration_steps:
            return True

        from alto.modifiers.lpt.adahop_internals.pattern_aggregation import (aggregate_patterns, patterns_to_modes)
        logger.info(f"[AdaHOP] Calibration window complete. Aggregating {self._step_idx} steps "
                    f"of observations across {total_layers} layers.")
        aggregated = aggregate_patterns(self._per_step_patterns)
        logger.info(f"[AdaHOP] Aggregation done. Mapping pattern pairs → TransformModes "
                    f"via recipe table ({len(self.layer_transform_config)} entries).")
        modes_by_fqn = patterns_to_modes(aggregated, self.layer_transform_config)
        self._log_mode_table(modes_by_fqn, aggregated=aggregated)

        effective_dump_path = self.dump_json_path or "./outputs/adahop_calibration.json"
        write_modes_json(effective_dump_path, aggregated, modes_by_fqn)
        logger.info(f"[AdaHOP] JSON dump: wrote {len(modes_by_fqn)} layer modes to {effective_dump_path}")

        logger.info("[AdaHOP] Phase B (in-place mode application) starting...")
        self._apply_modes_in_place(modes_by_fqn)
        self._detach_observation_hooks()
        # Record into the checkpointable calibration state so the next checkpoint
        # carries the modes and a resume can skip calibration.
        set_calibration_result(modes_by_fqn, self._step_idx)
        logger.info(f"[AdaHOP] Phase B complete. Modes set in place for {len(modes_by_fqn)} layers; "
                    f"recorded in CalibrationStateManager. Observation hooks detached.")
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        logger.info(f"[AdaHOP] Modifier finalize: phase_b_done={self._phase_b_done}, "
                    f"total_calibration_steps_seen={self._step_idx}.")
        self._detach_observation_hooks()
        return True

    def _log_mode_table(
        self,
        modes_by_fqn: Dict[str, Dict[str, str]],
        aggregated: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        """Emit a compact per-layer ``x w g -> forward_y backward_gx backward_gw`` table to stdout."""
        logger.info("[AdaHOP] Per-layer mode assignments:")
        for fqn in sorted(modes_by_fqn.keys()):
            slots = modes_by_fqn[fqn]
            if aggregated and fqn in aggregated:
                pats = aggregated[fqn]
                prefix = (f"x={pats.get('x', '?')} "
                          f"w={pats.get('w', '?')} "
                          f"g={pats.get('grad_output', '?')} -> ")
            else:
                prefix = ""
            logger.info(f"  {fqn}: {prefix}"
                        f"forward_y={slots.get('forward_y', 'none')} "
                        f"backward_gx={slots.get('backward_gx', 'none')} "
                        f"backward_gw={slots.get('backward_gw', 'none')}")

    # ------------------------------------------------------------------ helpers

    def _arm_calibration(self) -> None:
        """Register transient module observation hooks and begin Phase A."""
        if self._calibration_armed:
            return
        from alto._adahop_bridge import detect_outlier_pattern

        for fqn, module in self._fqn_to_module.items():
            h_fwd = module.register_forward_pre_hook(
                make_forward_pre_hook(self, fqn, detect_outlier_pattern)
            )
            h_bwd = module.register_full_backward_hook(
                make_backward_hook(self, fqn, detect_outlier_pattern)
            )
            self._handles.append(h_fwd)
            self._handles.append(h_bwd)
        self._calibration_armed = True
        logger.info(f"[AdaHOP] Calibration armed: {len(self._fqn_to_module)} layers, "
                    f"{self.calibration_steps} steps. Module forward-pre + backward hooks registered.")

    def _make_ht_resolver(self) -> Callable[[torch.device], Any]:
        if not self.use_hadamard:
            return lambda _dev: None
        from alto._adahop_bridge import HadamardFactory
        ht_cache: Dict[torch.device, Any] = {}

        def _get_ht(device):
            if device not in ht_cache:
                ht_cache[device] = HadamardFactory.create_transform(device=device)
            return ht_cache[device]

        return _get_ht

    def _collect_wrappers(self, model_parts, wrapper_cls) -> None:
        # Attach to the canonical instance F.linear will see (see long comment
        # below). Same discovery logic as before; only the target class changed
        # to the single MXFP4AdaHOPWrapper.
        #
        # No-FSDP path: `module.weight` IS the subclass. FSDP path: the
        # canonical instance lives at fsdp_param._sharded_local_tensor and is
        # the one FSDP later passes into fsdp_post_all_gather; in-place mode
        # updates there propagate to the unsharded param used in forward.
        from torch.distributed.tensor import DTensor
        try:
            from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
        except Exception:
            _get_module_fsdp_state = lambda _m: None  # noqa: E731

        self._fqn_to_wrapper.clear()
        self._fqn_to_module.clear()

        module_param_to_fsdp_param = {}
        for part in model_parts:
            for _m in part.modules():
                state = _get_module_fsdp_state(_m)
                if state is None or state._fsdp_param_group is None:
                    continue
                for fp in state._fsdp_param_group.fsdp_params:
                    mi = getattr(fp, "_module_info", None)
                    if mi is None:
                        continue
                    module_param_to_fsdp_param[(id(mi.module), mi.param_name)] = fp

        for part in model_parts:
            for module_fqn, module in part.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue

                fp = module_param_to_fsdp_param.get((id(module), "weight"))
                if fp is not None:
                    inner = fp._sharded_local_tensor
                elif isinstance(weight, wrapper_cls):
                    inner = weight
                else:
                    raw = weight.data
                    if isinstance(raw, DTensor):
                        inner = raw._local_tensor
                    else:
                        inner = raw

                if not isinstance(inner, wrapper_cls):
                    continue
                self._fqn_to_wrapper[module_fqn] = inner
                self._fqn_to_module[module_fqn] = module

    def _apply_modes_in_place(self, modes_by_fqn: Dict[str, Dict[str, str]]) -> None:
        """Set per-slot modes on every wrapper IN PLACE (no Parameter swap)."""
        ht_resolver = self._make_ht_resolver()
        n = 0
        for fqn, wrapper in self._fqn_to_wrapper.items():
            modes = modes_by_fqn.get(fqn, _NONE_MODES)
            ht = ht_resolver(wrapper._data.device)
            wrapper.set_modes(
                forward_y_mode=modes.get("forward_y", "none"),
                backward_gx_mode=modes.get("backward_gx", "none"),
                backward_gw_mode=modes.get("backward_gw", "none"),
                hadamard_transform=ht,
            )
            n += 1
        ht_state = "attached" if self.use_hadamard else "none"
        logger.info(f"[AdaHOP] Applied modes in place for {n} layers (hadamard={ht_state}).")
        self._phase_b_done = True

    def _detach_observation_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._calibration_armed = False
