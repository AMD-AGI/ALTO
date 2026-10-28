# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Literal, Iterable, TYPE_CHECKING
from torch.nn import Module
from compressed_tensors.utils import match_named_modules, match_name
from pydantic import PrivateAttr, Field, field_validator, model_validator
from alto.modifiers import Modifier
from alto.kernels.dispatch import TrainingOpConfig
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.nn.lpt import LPTModule, apply_lpt_module_configs
from alto.components.optimizer import DeOscillationConfig, enable_de_oscillation
import logging

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from torchtitan.models.base import BaseModel

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
    use_uos: bool = False

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
    def validate_uos(self):
        if not self.use_uos:
            return self
        schemes = self.scheme if isinstance(self.scheme, dict) else {self.scheme: None}
        if any(scheme_name != "mxfp4" for scheme_name in schemes):
            raise ValueError("UOS is only supported for mxfp4")
        if self.clip_mode != "none":
            raise ValueError("UOS is only supported with clip mode none")
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
                    use_uos=self.use_uos,
                )
                self._resolved_config[scheme_obj] = targets
        return self._resolved_config

    def on_convert(self, model: Module, **kwargs) -> bool:
        for _scheme_obj, targets in self.resolved_config.items():
            for name, module in match_named_modules(model, targets, self.ignore):
                if isinstance(module, LPTModule):
                    continue
                else:
                    raise ValueError(
                        f"{name}: expected an initialized LPT module, got {type(module)}"
                    )

        logger.info(f'LowPrecisionTrainingModifier converted model: {model}')
        print(f"converted model: {model}")
        return True

    def on_convert_config(self, model_config: "BaseModel.Config") -> bool:
        # convert configs to enable token alignment
        from torchtitan.models.common.moe import RoutedExperts
        from torchtitan.quantization.utils import swap_token_dispatcher

        for _fqn, config, _parent, _attr in model_config.traverse(RoutedExperts.Config):
            swap_token_dispatcher(config, ALIGN_SIZE_M)

        def is_match(
            name: str,
            module_cls: type,
            targets: str | Iterable[str],
            ignore: str | Iterable[str] = tuple(),
        ) -> bool:
            targets = [targets] if isinstance(targets, str) else targets
            ignore = [ignore] if isinstance(ignore, str) else ignore
            # Same semantics as compressed_tensors: any class in the MRO may match.
            cls_names = {c.__name__ for c in module_cls.__mro__}

            return any(
                match_name(name, target) or target in cls_names
                for target in targets
            ) and not any(
                match_name(name, ign) or ign in cls_names for ign in ignore
            )

        apply_lpt_module_configs(
            model_config,
            list(self.resolved_config.items()),
            lambda fqn, owner, targets: is_match(fqn, owner, targets, self.ignore),
        )
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        trainer = kwargs.get("trainer", None)

        if self.deosc_step > 0:
            if trainer is None:
                raise ValueError("Trainer must be passed to the pre_step method to enable weight de-oscillation.")
            current_step = trainer.engine.num_completed_steps + 1
            if current_step >= self.deosc_step:
                deosc_config = DeOscillationConfig(
                    enable=True,
                    period=self.deosc_period,
                    ratio_threshold=self.deosc_ratio,
                    log_freq=self.deosc_log_freq,
                )
                enable_de_oscillation(
                    trainer.engine.optim.optimizers,
                    deosc_config,
                    model_parts,
                )
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
