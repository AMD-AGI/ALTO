# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal, Iterable, TYPE_CHECKING
import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules, match_name
from pydantic import PrivateAttr, Field, field_validator, model_validator
from torchtitan.models.common.attention import BaseAttention, ScaledDotProductAttention
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.kernels.dispatch import (
    swap_params,
    TrainingOpConfig,
    LPScaledDotProductAttention,
)
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.nn import DecomposedLinear
from alto.components.optimizer import DeOscillationConfig, enable_de_oscillation

if TYPE_CHECKING:
    from torchtitan.protocols.model import BaseModel

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

    @field_validator("targets", mode="before")
    def validate_targets(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("scheme", mode="before")
    def validate_scheme(cls, value: str | dict[str, str | list[str]]) -> str | dict[str, list[str]]:
        if isinstance(value, str) and value not in ["mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "nvfp4", "amdfp4"]:
            raise ValueError(f"Unsupported training op scheme: {value}")

        if isinstance(value, dict):
            for scheme_name in value:
                cls.validate_scheme(scheme_name)

            for key, target in value.items():
                value[key] = cls.validate_targets(target)

        return value

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
                    raise ValueError(
                        f"lora_rank must be divisible by 32 for {scheme_name}, got {self.lora_rank}"
                    )
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
                    assert isinstance(module.inner_attention, ScaledDotProductAttention), "Only SDPA attention is supported for now."
                    module.inner_attention = LPScaledDotProductAttention(config=scheme_obj)
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

    def on_convert_config(self, model_config: "BaseModel.Config") -> bool:
        # convert configs to enable token alignment
        from torchtitan.models.common.moe import GroupedExperts
        from torchtitan.components.quantization.utils import swap_token_dispatcher
        from torchtitan.models.common.nn_modules import Linear

        for _fqn, config, parent, attr in model_config.traverse(GroupedExperts.Config):
            swap_token_dispatcher(config, ALIGN_SIZE_M)

        def is_match(
            name: str,
            module_cls_name: str,
            targets: str | Iterable[str],
            ignore: str | Iterable[str] = tuple(),
        ) -> bool:
            targets = [targets] if isinstance(targets, str) else targets
            ignore = [ignore] if isinstance(ignore, str) else ignore

            return any(
                match_name(name, target) or module_cls_name == target
                for target in targets
            ) and not any(
                match_name(name, ign) or module_cls_name == ign for ign in ignore
            )

        # replace Linear with DecomposedLinear
        if self.lora_rank > 0:
            resolved_targets = list(self.resolved_config.values())
            for _fqn, config, parent, attr in model_config.traverse(Linear.Config):
                for target in resolved_targets:
                    if is_match(_fqn, "Linear", target, self.ignore):
                        new_config = DecomposedLinear.Config(
                            in_features=config.in_features,
                            out_features=config.out_features,
                            bias=config.bias,
                            lora_rank=config.lora_rank,
                            param_init=config.param_init | DecomposedLinear._EXTRA_INIT,
                        )
                        setattr(parent, attr, new_config)
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
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
