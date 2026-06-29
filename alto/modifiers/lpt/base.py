# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Any, Literal
import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from pydantic import PrivateAttr, Field, field_validator, model_validator
from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.kernels.dispatch import (
    swap_params,
    TrainingOpConfig,
    LPScaledDotProductAttentionWrapper,
)
from alto.kernels.dispatch.config import PrecisionScheduleEntry
from alto.kernels.dispatch.tensor import set_training_precision_schedule_step
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.nn import DecomposedLinear

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
    mxfp4_scale_selection: Literal["default", "mse_4_6_shifted", "mse_4_6_strict"] = "default"
    precision_schedule: list[dict[str, Any]] = Field(default_factory=list)

    lora_rank: int = 0
    """
    Lora rank for the decomposed linear layer.
    If 0, use the original linear layer.
    """

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

    @staticmethod
    def _normalize_precision_name(precision: str) -> str:
        if precision == "mxfp8":
            return "mxfp8_e4m3"
        if precision not in ("bf16", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4"):
            raise ValueError(f"Unsupported scheduled precision: {precision}")
        return precision

    @field_validator("precision_schedule", mode="before")
    def validate_precision_schedule(cls, value: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("precision_schedule must be a list")

        normalized = []
        for idx, entry in enumerate(value):
            if not isinstance(entry, dict):
                raise ValueError(f"precision_schedule[{idx}] must be a mapping")
            if "precision" not in entry or "start_step" not in entry:
                raise ValueError(f"precision_schedule[{idx}] requires precision and start_step")

            start_step = int(entry["start_step"])
            end_step = entry.get("end_step", None)
            end_step = None if end_step is None else int(end_step)
            if start_step < 1:
                raise ValueError(f"precision_schedule[{idx}].start_step must be >= 1")
            if end_step is not None and end_step <= start_step:
                raise ValueError(f"precision_schedule[{idx}].end_step must be greater than start_step")

            normalized.append({
                "precision": cls._normalize_precision_name(str(entry["precision"])),
                "start_step": start_step,
                "end_step": end_step,
            })
        return normalized

    @model_validator(mode="after")
    def validate_lora_rank_alignment(self):
        if self.lora_rank <= 0:
            return self

        schemes = self.scheme if isinstance(self.scheme, dict) else {self.scheme: None}
        for scheme_name in schemes:
            if scheme_name == "nvfp4":
                if self.lora_rank % 16 != 0:
                    raise ValueError(
                        f"lora_rank must be divisible by 16 for nvfp4, got {self.lora_rank}"
                    )
            elif scheme_name in ("mxfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
                if self.lora_rank % 32 != 0:
                    raise ValueError(
                        f"lora_rank must be divisible by 32 for {scheme_name}, got {self.lora_rank}"
                    )
        return self

    @model_validator(mode="after")
    def validate_precision_schedule_support(self):
        if not self.precision_schedule:
            return self

        schemes = self.scheme if isinstance(self.scheme, dict) else {self.scheme: None}
        if any(precision in ("mxfp8_e4m3", "mxfp8_e5m2") for precision in schemes):
            scheduled_precisions = {entry["precision"] for entry in self.precision_schedule}
            if "nvfp4" in scheduled_precisions:
                raise ValueError("MXFP8 wrapper does not support scheduled nvfp4. Use mxfp4/nvfp4 as base scheme.")
        return self

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
                    mxfp4_scale_selection=self.mxfp4_scale_selection,
                    precision_schedule=tuple(
                        PrecisionScheduleEntry(**entry) for entry in self.precision_schedule
                    ),
                )
                self._resolved_config[scheme_obj] = targets
        return self._resolved_config

    def on_convert(self, model: Module, **kwargs) -> bool:
        for scheme_obj, targets in self.resolved_config.items():
            for name, module in match_named_modules(model, targets, self.ignore):
                if isinstance(module, BaseAttention):
                    assert module.attn_backend == "sdpa", "Only SDPA attention is supported for now."
                    module.inner_attention = LPScaledDotProductAttentionWrapper(config=scheme_obj)
                elif isinstance(module, torch.nn.Linear):
                    if self.lora_rank > 0:
                        module = DecomposedLinear.from_linear(module, lora_rank=self.lora_rank)
                        swap_params(module, config=scheme_obj, target_parameter_name="weight")
                        swap_params(module, config=scheme_obj, target_parameter_name="u")
                        swap_params(module, config=scheme_obj, target_parameter_name="v")
                        model.set_submodule(name, module, strict=True)
                    else:
                        if 32 in tuple(module.weight.shape):
                            logger.info(
                                f"Skipping {name} for {scheme_obj.precision} because weight shape is {tuple(module.weight.shape)}"
                            )
                            continue
                        swap_params(
                            module,
                            config=scheme_obj,
                            module_name=name,
                            target_parameter_name="weight",
                        )
                elif module.__class__.__name__.endswith("GroupedExperts"):
                    swap_params(module, config=scheme_obj, module_name=name)
                else:
                    raise ValueError(f"Unsupported module type: {type(module)}")

        logger.info(f"LowPrecisionTrainingModifier converted model: {model}")
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        for model_part in model_parts:
            for child in model_part.modules():
                if isinstance(child, DecomposedLinear):
                    child.init_lora_weights(init_std=0.02)
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        step = kwargs.get("step", None)
        if step is not None:
            set_training_precision_schedule_step(int(step))
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
