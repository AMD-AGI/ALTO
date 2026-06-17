# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional

from pydantic import Field, PrivateAttr
from torch import nn
from compressed_tensors.utils import match_named_modules

from alto.modifiers import Modifier
from alto.kernels.fp4.fp4_common.grad_clip_config import GradClipConfig
from alto.kernels.fp4.fp4_common import grad_clip_registry

__all__ = ["GradientClippingModifier"]


class GradientClippingModifier(Modifier):
    """Injects per-layer gradient clipping into MXFP4LinearFunction.backward.

    Must be listed **after** LowPrecisionTrainingModifier in the recipe YAML so
    that swap_params has already run when on_initialize sets module_id on each
    wrapped weight tensor.

    Two clip sites per MXFP4 nn.Linear backward:
      - grad_output before convert_to_mxfp4 (controls quantizer input scale)
      - grad_weights after the wgrad GEMM (limits per-layer optimizer step size)
    """

    targets: list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=lambda: ["output", "re:.*\\.router\\.gate"])

    clip_grad_output: bool = True
    grad_output_max_norm: Optional[float] = None
    grad_output_clip_value: Optional[float] = None

    clip_grad_weight: bool = True
    grad_weight_max_norm: Optional[float] = None
    grad_weight_clip_value: Optional[float] = None

    _registered_ids: list[int] = PrivateAttr(default_factory=list)

    def on_convert(self, model, **kwargs) -> bool:
        return True

    def on_initialize(self, model_parts: list[nn.Module], **kwargs) -> bool:
        cfg = GradClipConfig(
            clip_grad_output=self.clip_grad_output,
            grad_output_max_norm=self.grad_output_max_norm,
            grad_output_clip_value=self.grad_output_clip_value,
            clip_grad_weight=self.clip_grad_weight,
            grad_weight_max_norm=self.grad_weight_max_norm,
            grad_weight_clip_value=self.grad_weight_clip_value,
        )
        for model_part in model_parts:
            for _fqn, module in match_named_modules(model_part, self.targets, self.ignore):
                mid = id(module)
                grad_clip_registry.register(mid, cfg)
                self._registered_ids.append(mid)
                # Wire the module_id onto the wrapped weight tensor so that
                # MXFP4LinearFunction.backward can look up cfg via the registry.
                if hasattr(module, "weight") and hasattr(module.weight, "module_id"):
                    module.weight.module_id = mid
        return True

    def on_finalize(self, model_parts: list[nn.Module], **kwargs) -> bool:
        for mid in self._registered_ids:
            grad_clip_registry.deregister(mid)
        self._registered_ids.clear()
        return True

    def on_pre_step(self, model_parts: list[nn.Module], **kwargs) -> bool:
        return True

    def on_post_step(self, model_parts: list[nn.Module], **kwargs) -> bool:
        return True
