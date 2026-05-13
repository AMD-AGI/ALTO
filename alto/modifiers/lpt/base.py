# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal
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
    two_level_scaling: Literal["none", "tensorwise", "blockwise", "outer_block"] = "none"
    clip_mode: Literal["none", "static", "dynamic"] = "none"
    use_outer_2dblock_x: bool = False
    use_outer_2dblock_w: bool = False
    outer_block_size: int = 128

    # Paper-recipe knobs for TetraJet-v2 (arXiv 2510.27527) NVFP4 linear:
    #   - align_x_forward_wgrad: reuse forward X̂ in wgrad-axis QDQ
    #     (Tab. 8(b) "X̂ align" → -0.31 PPL on OLMo-2-150M / 52B tokens).
    #   - use_dx_rht: N-axis RHT in dX backward
    #     (Tab. 8(c) "dX RHT" → -0.05 PPL on top of dW RHT).
    align_x_forward_wgrad: bool = False
    use_dx_rht: bool = False

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

    @model_validator(mode="after")
    def validate_outer_block_scaling(self):
        if self.outer_block_size < 16 or self.outer_block_size % 16 != 0:
            raise ValueError(
                "outer_block_size must be a positive multiple of the NVFP4 "
                f"inner block size (16), got {self.outer_block_size}"
            )
        outer_fields_are_active = (
            self.use_outer_2dblock_x
            or self.use_outer_2dblock_w
            or self.outer_block_size != 128
        )
        if self.two_level_scaling != "outer_block" and outer_fields_are_active:
            raise ValueError(
                "use_outer_2dblock_x, use_outer_2dblock_w, and non-default "
                "outer_block_size are only meaningful when "
                'two_level_scaling="outer_block".'
            )
        if self.two_level_scaling == "outer_block":
            schemes = [self.scheme] if isinstance(self.scheme, str) else list(self.scheme)
            invalid_schemes = [scheme for scheme in schemes if scheme != "nvfp4"]
            if invalid_schemes:
                raise ValueError(
                    'two_level_scaling="outer_block" is only defined for '
                    f"scheme='nvfp4', got {invalid_schemes}"
                )
            # Outer-block scaling only supports the dense Linear path; the
            # NVFP4 grouped-GEMM dispatch raises ``NotImplementedError`` on
            # MoE forward.  Reject the recipe here so the failure points at
            # the offending yaml entry instead of a deep runtime crash.
            offending = self._collect_moe_like_targets()
            if offending:
                raise ValueError(
                    'two_level_scaling="outer_block" does not support MoE / '
                    f"grouped-experts modules; offending targets: {offending}. "
                    "Drop these from `targets` or set two_level_scaling to "
                    '"tensorwise" / "none".'
                )

        # Paper-recipe knob compatibility checks.
        if self.align_x_forward_wgrad:
            if self.two_level_scaling != "outer_block":
                raise ValueError(
                    'align_x_forward_wgrad=True requires '
                    'two_level_scaling="outer_block" (paper recipe).'
                )
            if self.use_2dblock_x:
                raise ValueError(
                    "align_x_forward_wgrad=True only makes sense with the "
                    "1D inner block on X (use_2dblock_x=False).  With 2D "
                    "inner blocks the fprop axis=-1 view is already aliased "
                    "to the wgrad axis=0 view."
                )
            if self.use_hadamard:
                raise ValueError(
                    "align_x_forward_wgrad=True is incompatible with "
                    "use_hadamard=True: M-axis Hadamard rotates X before the "
                    "axis=0 QDQ; reusing the unrotated forward X̂ would break "
                    "the (Hx)ᵀ(Hg) = xᵀg invariance.  Pick one."
                )
            if self.use_dge:
                raise ValueError(
                    "align_x_forward_wgrad=True is not supported with "
                    "use_dge=True (DGE relies on the wgrad-axis weight raw "
                    "FP4 view, not exercised on this code path)."
                )

        if self.use_dx_rht and self.two_level_scaling != "outer_block":
            raise ValueError(
                'use_dx_rht=True requires two_level_scaling="outer_block" '
                "(paper recipe; outside this scope the dX path has not been "
                "audited for invariance)."
            )
        return self

    def _collect_moe_like_targets(self) -> list[str]:
        """Return targets whose name looks like an MoE / GroupedExperts module.

        Match is a case-insensitive substring check on ``"groupedexperts"`` or
        ``"moe"``, so regex patterns such as ``re:.*GroupedExperts`` are
        matched literally without parsing them.
        """
        target_groups: list[list[str]] = []
        if isinstance(self.scheme, dict):
            target_groups.extend(list(self.scheme.values()))
        else:
            target_groups.append(
                self.targets if isinstance(self.targets, list) else [self.targets]
            )
        offending: list[str] = []
        for group in target_groups:
            for entry in group:
                lowered = entry.lower()
                if "groupedexperts" in lowered or "moe" in lowered:
                    offending.append(entry)
        return offending

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
                    use_outer_2dblock_x=self.use_outer_2dblock_x,
                    use_outer_2dblock_w=self.use_outer_2dblock_w,
                    outer_block_size=self.outer_block_size,
                    align_x_forward_wgrad=self.align_x_forward_wgrad,
                    use_dx_rht=self.use_dx_rht,
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
                        swap_params(module, config=scheme_obj, module_name=name)
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
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
