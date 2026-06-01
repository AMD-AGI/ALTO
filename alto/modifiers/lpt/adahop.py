# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""AdaHOP calibration + per-slot Hadamard mode selection modifier.

Orchestrates the two-phase swap described in ADAHOP_TO_ALTO_INTEGRATION_PLAN.md:

1. **Phase A** is set up by ``LowPrecisionTrainingModifier`` when ``scheme="mxfp4_adahop"``
   is used in the recipe — all targeted weights wrapped in
   :class:`MXFP4CalibrationWrapper`. Identical dispatch to plain MXFP4.
2. **Phase A → Phase B transition** is owned by this modifier:

   * ``on_initialize`` walks the model, finds every ``MXFP4CalibrationWrapper``
     weight, attaches an outlier-pattern observation callback (forward path),
     and registers a backward hook on the parent linear module for
     ``grad_output`` observation. The layer FQN is closed over in both
     closures so the wrapper itself stays FQN-agnostic.
   * ``on_pre_step`` opens a fresh per-step pattern dict and increments the
     step counter.
   * ``on_post_step`` does nothing until step == ``calibration_steps``, then:
     aggregates per-layer majority patterns, maps pattern pairs to per-slot
     ``TransformMode`` triples via the recipe's ``layer_transform_config``,
     and re-swaps every calibration wrapper as
     :class:`MXFP4AdaHOPWrapper` carrying the frozen modes. Observation
     hooks are removed.

A ``transform_config_path`` shortcut accepts a pre-baked JSON
(``{layer_fqn: {forward_y, backward_gx, backward_gw}}``) and skips
calibration entirely — the re-swap happens at ``on_initialize`` instead.
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
    make_forward_callback,
    write_modes_json,
)

__all__ = ["AdaHOPModifier"]


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
    _backward_handles: List[Any] = PrivateAttr(default_factory=list)
    _hadamard_transform: Any = PrivateAttr(default=None)
    _phase_b_done: bool = PrivateAttr(default=False)

    @property
    def requires_training_mode(self) -> bool:
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        # The Phase-A wrapper swap is performed by LowPrecisionTrainingModifier
        # when scheme="mxfp4_adahop". Configure the HadamardFactory here so
        # the transform object is available at re-swap time.
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
        from alto.kernels.dispatch.adahop_tensor import MXFP4CalibrationWrapper

        self._collect_calibration_wrappers(model_parts, MXFP4CalibrationWrapper)
        n_wrappers = len(self._fqn_to_wrapper)
        logger.info(f"[AdaHOP] Phase A: discovered {n_wrappers} MXFP4CalibrationWrapper-wrapped linears.")
        if n_wrappers > 0:
            sample = list(self._fqn_to_wrapper.keys())[:5]
            logger.info(f"[AdaHOP] First {len(sample)} FQNs: {', '.join(sample)}"
                        f"{' ...' if n_wrappers > 5 else ''}")

        if self.transform_config_path is not None:
            modes_by_fqn = load_modes_from_json(self.transform_config_path)
            logger.info(f"[AdaHOP] Phase B (JSON-load): {len(modes_by_fqn)} modes loaded from "
                        f"{self.transform_config_path}; re-swap starting (calibration skipped).")
            self._log_mode_table(modes_by_fqn, aggregated=None)
            self._do_phase_b_reswap(modes_by_fqn)
            return True

        if not self._fqn_to_wrapper:
            logger.warning("[AdaHOP] No MXFP4CalibrationWrapper found; "
                           "is LowPrecisionTrainingModifier(scheme='mxfp4_adahop') in the recipe?")
            return True

        from alto._adahop_bridge import detect_outlier_pattern

        for fqn, wrapper in self._fqn_to_wrapper.items():
            wrapper.attach_calibration_callback(make_forward_callback(self, fqn, detect_outlier_pattern))

        for fqn, module in self._fqn_to_module.items():
            handle = module.register_full_backward_hook(make_backward_hook(self, fqn, detect_outlier_pattern))
            self._backward_handles.append(handle)

        logger.info(f"[AdaHOP] Calibration armed: {n_wrappers} layers, {self.calibration_steps} steps. "
                    f"Forward callbacks attached. Backward hooks registered. "
                    f"Patterns will be observed every step.")
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.enabled or self._phase_b_done:
            return True
        if self.transform_config_path is not None:
            return True
        if self._step_idx >= self.calibration_steps:
            return True
        # Open a fresh dict for this step's pattern observations.
        self._per_step_patterns.append({})
        logger.info(f"[AdaHOP] Calibration step {self._step_idx + 1}/{self.calibration_steps} opening bucket.")
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.enabled or self._phase_b_done:
            return True
        if self.transform_config_path is not None:
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

        logger.info("[AdaHOP] Phase B re-swap starting...")
        self._do_phase_b_reswap(modes_by_fqn)
        self._detach_observation_hooks()
        logger.info(f"[AdaHOP] Phase B complete. Forward path now uses MXFP4AdaHOPLinearFunction "
                    f"with frozen modes for {len(modes_by_fqn)} layers. Observation hooks detached.")
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

    def _collect_calibration_wrappers(self, model_parts, cal_wrapper_cls) -> None:
        from torch.distributed.tensor import DTensor
        self._fqn_to_wrapper.clear()
        self._fqn_to_module.clear()
        for part in model_parts:
            for module_fqn, module in part.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                inner = weight.data
                if isinstance(inner, DTensor):
                    inner = inner._local_tensor
                if not isinstance(inner, cal_wrapper_cls):
                    continue
                self._fqn_to_wrapper[module_fqn] = inner
                self._fqn_to_module[module_fqn] = module

    def _do_phase_b_reswap(self, modes_by_fqn: Dict[str, Dict[str, str]]) -> None:
        from alto.kernels.dispatch.adahop_tensor import MXFP4AdaHOPWrapper

        ht = None
        if self.use_hadamard:
            from alto._adahop_bridge import HadamardFactory
            # Lazy construction; the modifier doesn't know the device until
            # weights actually live somewhere, so build per-device on demand.
            ht_cache: Dict[torch.device, Any] = {}

            def _get_ht(device):
                if device not in ht_cache:
                    ht_cache[device] = HadamardFactory.create_transform(device=device)
                return ht_cache[device]

            ht_resolver: Callable[[torch.device], Any] = _get_ht
        else:
            ht_resolver = lambda _dev: None  # noqa: E731

        from torch.distributed.tensor import DTensor

        for fqn, wrapper in self._fqn_to_wrapper.items():
            modes = modes_by_fqn.get(fqn, {"forward_y": "none", "backward_gx": "none", "backward_gw": "none"})
            module = self._fqn_to_module[fqn]
            old_param = module.weight
            data = wrapper._data
            ht = ht_resolver(data.device)
            new_wrapper = MXFP4AdaHOPWrapper(
                data,
                wrapper.config,
                hadamard_transform=ht,
                forward_y_mode=modes.get("forward_y", "none"),
                backward_gx_mode=modes.get("backward_gx", "none"),
                backward_gw_mode=modes.get("backward_gw", "none"),
            )
            if isinstance(old_param.data, DTensor):
                # Preserve FSDP sharding: rebuild the DTensor around the new local wrapper
                # using the same device mesh and placements as the existing param.
                old_dt = old_param.data
                new_dt = DTensor.from_local(
                    new_wrapper,
                    device_mesh=old_dt.device_mesh,
                    placements=old_dt.placements,
                    run_check=False,
                    shape=old_dt.shape,
                    stride=old_dt.stride(),
                )
                module.weight = nn.Parameter(new_dt, requires_grad=old_param.requires_grad)
            else:
                module.weight = nn.Parameter(new_wrapper, requires_grad=old_param.requires_grad)
            ht_state = "attached" if ht is not None else "none"
            logger.info(f"  [AdaHOP] re-swapped {fqn}: MXFP4CalibrationWrapper → "
                        f"MXFP4AdaHOPWrapper(forward_y={new_wrapper._forward_y_mode}, "
                        f"backward_gx={new_wrapper._backward_gx_mode}, "
                        f"backward_gw={new_wrapper._backward_gw_mode}, "
                        f"hadamard={ht_state})")

        self._phase_b_done = True

    def _detach_observation_hooks(self) -> None:
        for handle in self._backward_handles:
            handle.remove()
        self._backward_handles.clear()
        for wrapper in self._fqn_to_wrapper.values():
            if hasattr(wrapper, "attach_calibration_callback"):
                wrapper.attach_calibration_callback(None)
