# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from pydantic import PrivateAttr, Field, field_validator
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.kernels.dispatch import (
    swap_params,
    TrainingOpConfig,
    LPScaledDotProductAttentionWrapper,
)
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M

__all__ = ["LowPrecisionTrainingModifier"]


class LowPrecisionTrainingModifier(Modifier):

    scheme: str | dict[str, list[str]]
    targets: str | list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=lambda: ["output"])

    use_2dblock_x: bool = False
    use_2dblock_w: bool = True
    use_hadamard: bool = False
    use_sr_grad: bool = False
    use_dge: bool = False
    two_level_scaling: Literal["none", "tensorwise", "blockwise"] = "none"
    clip_mode: Literal["none", "static", "dynamic"] = "none"

    _resolved_config: dict[TrainingOpConfig, list[str]] | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        set_token_group_alignment_size_m(ALIGN_SIZE_M)

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("scheme", mode="before")
    def validate_scheme(cls, value: str | dict[str, str | list[str]]) -> str | dict[str, list[str]]:
        if isinstance(value, str) and value not in ["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"]:
            raise ValueError(f"Unsupported training op scheme: {value}")

        if isinstance(value, dict):
            for scheme_name in value:
                cls.validate_scheme(scheme_name)

            for key, target in value.items():
                value[key] = cls.validate_targets(target)

        return value

    @property
    def requires_training_mode(self) -> bool:
        return True

    @property
    def resolved_config(self) -> dict[TrainingOpConfig, list[str]]:
        if self._resolved_config is None:
            # if target is provided with scheme name
            if isinstance(self.scheme, str):
                self.scheme = {self.scheme: self.targets}

            self._resolved_config = {}
            for scheme_name, targets in self.scheme.items():
                scheme_obj = TrainingOpConfig(
                    precision=scheme_name,
                    use_2dblock_x=self.use_2dblock_x,
                    use_2dblock_w=self.use_2dblock_w,
                    use_hadamard=self.use_hadamard,
                    use_sr_grad=self.use_sr_grad,
                    use_dge=self.use_dge,
                    two_level_scaling=self.two_level_scaling,
                    clip_mode=self.clip_mode,
                )
                self._resolved_config[scheme_obj] = targets
        return self._resolved_config

    def on_convert(self, model: Module, **kwargs) -> bool:
        for scheme_obj, targets in self.resolved_config.items():
            for name, module in match_named_modules(model, targets, self.ignore):
                if isinstance(module, BaseAttention):
                    assert module.attn_backend == "sdpa", "Only SDPA attention is supported for now."
                    module.inner_attention = LPScaledDotProductAttentionWrapper(config=scheme_obj)
                elif isinstance(module, torch.nn.Linear) or module.__class__.__name__.endswith("GroupedExperts"):
                    swap_params(module, config=scheme_obj, module_name=name)
                else:
                    raise ValueError(f"Unsupported module type: {type(module)}")

        logger.info(f"LowPrecisionTrainingModifier converted model: {model}")
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
