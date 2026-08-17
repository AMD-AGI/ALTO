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
from alto.components.optimizer import DeOscillationConfig, enable_de_oscillation

__all__ = ["LowPrecisionTrainingModifier"]


class LowPrecisionTrainingModifier(Modifier):

    scheme: str | dict[str, list[str]]
    targets: str | list[str] = Field(default_factory=lambda: ["Linear"])
    ignore: list[str] = Field(default_factory=lambda: ["output"])

    use_2dblock_x: bool = False
    use_2dblock_w: bool = True
    use_hadamard: bool = False
    hadamard_type: Literal["default", "3rht"] = "default"
    use_sr_grad: bool = False
    use_dge: bool = False
    full_precision_backward: bool = False
    """
    Quantize the forward only; keep the backward (dgrad + wgrad) in bf16 with the
    gradient unquantized. Dense MXFP4 linear path only.
    """
    two_level_scaling: Literal["none", "tensorwise", "blockwise"] = "none"
    clip_mode: Literal["none", "static", "dynamic"] = "none"
    blockscale_selection: Literal["default", "midmax-legacy", "uos", "uos6"] = "default"


    lora_rank: int = 0
    """
    Lora rank for the decomposed linear layer.
    If 0, use the original linear layer.
    """

    deosc_step: int = 0
    """
    Step to enable weight de-oscillation.
    If 0, weight de-oscillation is disabled.
    """

    deosc_period: int = 200
    """
    De-oscillation observation/update period.
    """

    deosc_ratio: float = 4.0
    """
    Score threshold to reset weights in de-oscillation.
    """

    deosc_log_freq: int = 1
    """
    Log frequency to report de-oscillation statistics.
    If 0, weight de-oscillation statistics logging is disabled.
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
        if isinstance(value, str) and value not in ["mxfp4", "mxfp4_adahop", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4", "amdfp4"]:
            raise ValueError(f"Unsupported training op scheme: {value}")

        if isinstance(value, dict):
            for scheme_name in value:
                cls.validate_scheme(scheme_name)

            for key, target in value.items():
                value[key] = cls.validate_targets(target)

        return value

    @model_validator(mode="after")
    def validate_full_precision_backward(self):
        if self.full_precision_backward:
            scheme_names = self.scheme if isinstance(self.scheme, dict) else {self.scheme: None}
            for scheme_name in scheme_names:
                if scheme_name not in ("mxfp4",):
                    raise ValueError(
                        f"full_precision_backward is only supported for scheme 'mxfp4', got '{scheme_name}'")
            if self.use_dge:
                raise ValueError("full_precision_backward is incompatible with use_dge "
                                 "(DGE is a gradient-quantization estimator; the backward is not quantized here)")
        return self

    @model_validator(mode="after")
    def validate_lora_rank_alignment(self):
        if self.lora_rank <= 0:
            return self

        schemes = self.scheme if isinstance(self.scheme, dict) else {self.scheme: None}
        for scheme_name in schemes:
            if scheme_name in ("nvfp4", "amdfp4"):
                if self.lora_rank % 16 != 0:
                    raise ValueError(
                        f"lora_rank must be divisible by 16 for {scheme_name}, got {self.lora_rank}"
                    )
            elif scheme_name in ("mxfp4", "mxfp8_e4m3", "mxfp8_e5m2"):
                if self.lora_rank % 32 != 0:
                    raise ValueError(f"lora_rank must be divisible by 32 for {scheme_name}, got {self.lora_rank}")
        return self

    @field_validator("deosc_step", mode="after")
    def validate_deosc_step(cls, value: int) -> int:
        if value < 0:
            raise ValueError("deosc_step must be non-negative")
        return value
    
    @field_validator("deosc_period", mode="after")
    def validate_deosc_period(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("deosc_period must be positive")
        return value
    
    @field_validator("deosc_ratio", mode="after")
    def validate_deosc_ratio(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("deosc_ratio must be positive")
        return value
    
    @field_validator("deosc_log_freq", mode="after")
    def validate_deosc_log_freq(cls, value: int) -> int:
        if value < 0:
            raise ValueError("deosc_log_freq must be non-negative")
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
                # "mxfp4_adahop" reuses the MXFP4 kernel path; the only
                # difference is which wrapper class is used at swap time
                # (handled in on_convert). At the TrainingOpConfig level it
                # is plain mxfp4.
                precision = "mxfp4" if scheme_name == "mxfp4_adahop" else scheme_name
                scheme_obj = TrainingOpConfig(
                    precision=precision,
                    use_2dblock_x=self.use_2dblock_x,
                    use_2dblock_w=self.use_2dblock_w,
                    use_hadamard=self.use_hadamard,
                    use_sr_grad=self.use_sr_grad,
                    use_dge=self.use_dge,
                    full_precision_backward=self.full_precision_backward,
                    two_level_scaling=self.two_level_scaling,
                    clip_mode=self.clip_mode,
                    blockscale_selection=self.blockscale_selection,
                )
                # Tag the underlying scheme so on_convert can pick the right
                # wrapper class. Stored on the dict key via a sibling attribute
                # of the modifier rather than mutating TrainingOpConfig.
                self._resolved_config[scheme_obj] = targets
                self._scheme_tag = getattr(self, "_scheme_tag", {})
                self._scheme_tag[scheme_obj] = scheme_name
        return self._resolved_config

    def on_convert(self, model: Module, **kwargs) -> bool:
        if self.use_hadamard and self.hadamard_type != "default":                                
          from alto.kernels.hadamard_transform import HadamardFactory
          HadamardFactory.configure(transform_type=self.hadamard_type)
        for scheme_obj, targets in self.resolved_config.items():
            tensor_cls = self._wrapper_cls_for_scheme(scheme_obj)
            scheme_name = getattr(self, "_scheme_tag", {}).get(scheme_obj, scheme_obj.precision)
            wrapper_label = tensor_cls.__name__ if tensor_cls is not None else "default-for-precision"
            logger.info(f"LowPrecisionTrainingModifier: scheme={scheme_name}, wrapper_cls={wrapper_label}, "
                        f"targets={targets}, ignore={self.ignore}")
            for name, module in match_named_modules(model, targets, self.ignore):
                if isinstance(module, BaseAttention):
                    assert module.attn_backend == "sdpa", "Only SDPA attention is supported for now."
                    module.inner_attention = LPScaledDotProductAttentionWrapper(config=scheme_obj)
                elif isinstance(module, torch.nn.Linear):
                    if self.lora_rank > 0:
                        module = DecomposedLinear.from_linear(module, lora_rank=self.lora_rank)
                        swap_params(module, config=scheme_obj, target_parameter_name="weight", tensor_cls=tensor_cls)
                        swap_params(module, config=scheme_obj, target_parameter_name="u", tensor_cls=tensor_cls)
                        swap_params(module, config=scheme_obj, target_parameter_name="v", tensor_cls=tensor_cls)
                        model.set_submodule(name, module, strict=True)
                    else:
                        swap_params(module, config=scheme_obj, module_name=name, tensor_cls=tensor_cls)
                elif module.__class__.__name__.endswith("GroupedExperts"):
                    swap_params(module, config=scheme_obj, module_name=name, tensor_cls=tensor_cls)
                else:
                    raise ValueError(f"Unsupported module type: {type(module)}")

        logger.info(f"LowPrecisionTrainingModifier converted model: {model}")
        return True

    def _wrapper_cls_for_scheme(self, scheme_obj):
        """Return the wrapper tensor class for ``scheme_obj``. ``None`` falls
        back to ``swap_params``' default (looked up from the config precision)."""
        scheme_name = getattr(self, "_scheme_tag", {}).get(scheme_obj)
        if scheme_name == "mxfp4_adahop":
            # Single-wrapper design: wrap with the AdaHOP wrapper from convert
            # time (modes default "none" == plain MXFP4 during calibration). The
            # AdaHOPModifier flips the modes in place at Phase B so they take
            # effect on the FSDP-owned tensor.
            from alto.kernels.dispatch.adahop_tensor import MXFP4AdaHOPWrapper
            return MXFP4AdaHOPWrapper
        return None

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        for model_part in model_parts:
            for child in model_part.modules():
                if isinstance(child, DecomposedLinear):
                    child.init_lora_weights(init_std=0.02)
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        trainer = kwargs.get("trainer", None)

        if self.deosc_step > 0:
            if trainer is None:
                raise ValueError("Trainer must be passed to the pre_step method to enable weight de-oscillation.")
            if trainer.step >= self.deosc_step:
                deosc_config = DeOscillationConfig(
                    enable=True,
                    period=self.deosc_period,
                    ratio_threshold=self.deosc_ratio,
                    log_freq=self.deosc_log_freq,
                )
                enable_de_oscillation(trainer.optimizers, deosc_config)
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
