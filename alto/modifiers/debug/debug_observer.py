# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""DebugObserverModifier — per-step tensor capture for low-precision MoE debugging.

Hooks into a normal training run, captures the raw tensors that flow through
targeted layers at every captured iteration, and dumps everything to a single
torch.save'd file at finalization.

See DEBUG_OBSERVER.md for the full output schema and quickstart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from pydantic import Field, PrivateAttr
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.modifiers.debug.observer_hooks import (
    make_linear_fwd_pre_hook,
    make_linear_bwd_hook,
    make_linear_grad_weight_hook,
    make_grouped_experts_fwd_pre_hook,
    make_grouped_experts_bwd_hook,
    make_grouped_experts_grad_weight_hook,
)

__all__ = ["DebugObserverModifier"]


class DebugObserverModifier(Modifier):
    """Captures input, weight, grad_output, and grad_weight tensors for targeted
    layers at configurable step intervals. Dumps to a .pt file on finalization.

    Composes with any other modifier (LPT, AdaHOP, BF16 baseline). Drop it into
    any recipe YAML under a ``debug_modifiers`` section.
    """

    targets: list[str] = Field(default_factory=lambda: ["Linear", "GptOssGroupedExperts"])
    ignore: list[str] = Field(default_factory=lambda: ["output", "re:.*\\.router\\.gate"])

    capture_every: int = 1
    max_captures: int = 10
    output_path: str = "./outputs/debug_obs.pt"

    capture_input: bool = True
    capture_weight: bool = True
    capture_grad_output: bool = True
    capture_grad_weight: bool = True

    # --- private mutable state ---
    _step_idx: int = PrivateAttr(default=0)
    # captures[fqn] = {"active": bool, step_idx: {"input": T, ...}, ...}
    _captures: dict = PrivateAttr(default_factory=dict)
    # number of steps for which we have already stored data
    _n_captured: int = PrivateAttr(default=0)
    # set to True once max_captures is reached and hooks are removed early
    _detached: bool = PrivateAttr(default=False)
    # one-element list shared with all hook closures so they always see the
    # current step without needing a reference to self (avoids Pydantic issues)
    _step_ref: list = PrivateAttr(default_factory=lambda: [0])
    # handles for parameter-level grad hooks (not managed by HooksMixin)
    _param_hook_handles: list = PrivateAttr(default_factory=list)

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        for model_part in model_parts:
            for fqn, module in match_named_modules(model_part, self.targets, self.ignore):
                self._captures[fqn] = {"active": False}
                cls_name = module.__class__.__name__

                if cls_name == "Linear" or isinstance(module, torch.nn.Linear):
                    self.register_hook(
                        module,
                        make_linear_fwd_pre_hook(
                            self._captures, fqn, self._step_ref,
                            self.capture_input, self.capture_weight,
                        ),
                        "forward_pre",
                    )
                    self.register_hook(
                        module,
                        make_linear_bwd_hook(
                            self._captures, fqn, self._step_ref, self.capture_grad_output,
                        ),
                        "full_backward",
                    )
                    if self.capture_grad_weight and hasattr(module, "weight") and module.weight is not None:
                        handle = module.weight.register_hook(
                            make_linear_grad_weight_hook(
                                self._captures, fqn, self._step_ref, self.capture_grad_weight,
                            )
                        )
                        self._param_hook_handles.append(handle)

                elif cls_name.endswith("GroupedExperts"):
                    self.register_hook(
                        module,
                        make_grouped_experts_fwd_pre_hook(
                            self._captures, fqn, self._step_ref,
                            self.capture_input, self.capture_weight,
                        ),
                        "forward_pre",
                    )
                    self.register_hook(
                        module,
                        make_grouped_experts_bwd_hook(
                            self._captures, fqn, self._step_ref, self.capture_grad_output,
                        ),
                        "full_backward",
                    )
                    if self.capture_grad_weight:
                        for attr, key in [
                            ("mlp1_weight", "grad_mlp1_weight"),
                            ("mlp2_weight", "grad_mlp2_weight"),
                        ]:
                            param = getattr(module, attr, None)
                            if param is not None:
                                handle = param.register_hook(
                                    make_grouped_experts_grad_weight_hook(
                                        self._captures, fqn, self._step_ref,
                                        key, self.capture_grad_weight,
                                    )
                                )
                                self._param_hook_handles.append(handle)

        logger.info(
            f"DebugObserverModifier: monitoring {len(self._captures)} layers, "
            f"capture_every={self.capture_every}, max_captures={self.max_captures}"
        )
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        self._step_idx += 1
        self._step_ref[0] = self._step_idx

        if self._detached:
            return True

        should_capture = (
            (self._step_idx % self.capture_every == 0) and
            (self._n_captured < self.max_captures)
        )
        for fqn in self._captures:
            self._captures[fqn]["active"] = should_capture

        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        if self._detached:
            return True

        # Check whether this step produced a capture
        step_idx = self._step_idx
        captured_this_step = any(
            step_idx in self._captures[fqn]
            for fqn in self._captures
        )
        if captured_this_step:
            self._n_captured += 1
            logger.debug(f"DebugObserverModifier: captured step {step_idx} ({self._n_captured}/{self.max_captures})")

        # Deactivate all gates
        for fqn in self._captures:
            self._captures[fqn]["active"] = False

        # Detach once max reached
        if self._n_captured >= self.max_captures:
            logger.info(f"DebugObserverModifier: reached max_captures={self.max_captures}, detaching hooks")
            self._detach()

        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        if not self._detached:
            self._detach()
        self._dump()
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detach(self) -> None:
        self.remove_hooks()
        for h in self._param_hook_handles:
            h.remove()
        self._param_hook_handles.clear()
        self._detached = True

    def _dump(self) -> None:
        rank = int(os.environ.get("RANK", 0))
        path = Path(self.output_path)
        if rank != 0:
            # Insert rank suffix: debug_obs.pt -> debug_obs_rank1.pt
            path = path.with_stem(f"{path.stem}_rank{rank}")

        path.parent.mkdir(parents=True, exist_ok=True)

        captured_steps = sorted({
            step
            for fqn, data in self._captures.items()
            for step in data
            if step != "active"
        })

        layer_shapes: dict[str, Any] = {}
        for fqn, data in self._captures.items():
            for step, tensors in data.items():
                if step == "active":
                    continue
                layer_shapes[fqn] = {k: list(t.shape) for k, t in tensors.items()}
                break

        blob: dict[str, Any] = {
            fqn: {step: tensors for step, tensors in data.items() if step != "active"}
            for fqn, data in self._captures.items()
        }
        blob["_meta"] = {
            "rank": rank,
            "iterations_captured": captured_steps,
            "layer_shapes": layer_shapes,
            "capture_every": self.capture_every,
            "max_captures": self.max_captures,
            "mlp2_input_captured": False,  # v1 limitation
        }

        torch.save(blob, path)
        logger.info(f"DebugObserverModifier: saved {len(captured_steps)} captures to {path}")
