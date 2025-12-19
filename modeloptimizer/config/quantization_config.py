import warnings
from typing import Optional, Literal
from copy import deepcopy
from dataclasses import dataclass, field
from torch import Tensor

from .registry import MODULE_QUANT_DEFAULTS


@dataclass
class QuantConfig:
    global_config: "ModuleQuantConfig"
    layer_config: dict[str, "ModuleQuantConfig"] = field(default_factory=dict)
    type_config: dict[str, "ModuleQuantConfig"] = field(default_factory=dict)

    def __post_init__(self):
        self.merge_module_defaults()

    def merge_module_defaults(self):
        for module_type_name, default_config in MODULE_QUANT_DEFAULTS.items():
            current_config = self.type_config.get(module_type_name, None)
            if current_config is None:
                current_config = deepcopy(self.global_config)
            else:
                current_config |= self.global_config
                pass
            self.type_config[module_type_name] = current_config | default_config

    def get_layer_config(self, name: str,
                         type_name: str) -> "ModuleQuantConfig":
        current_config = self.layer_config.get(name, None)
        if current_config is None:
            self.layer_config[name] = deepcopy(self.type_config[type_name])
        else:
            self.layer_config[name] |= self.type_config[type_name]
        return self.layer_config[name]


@dataclass
class ModuleQuantConfig:
    input: Optional["TensorQuantConfig"] = None
    inputx: Optional["TensorQuantConfig"] = None
    inputy: Optional["TensorQuantConfig"] = None
    weight: Optional["TensorQuantConfig"] = None
    bias: Optional["TensorQuantConfig"] = None
    output: Optional["TensorQuantConfig"] = None

    def __post_init__(self):
        if self.input is not None:
            if self.inputx is None:
                self.inputx = deepcopy(self.input)
            if self.inputy is None:
                self.inputy = deepcopy(self.input)

    def __or__(self, other: "ModuleQuantConfig"):
        # merge non-None entries from other into self
        for key, value in other.__dict__.items():
            if value is not None:
                self.__dict__[key] |= value
        return self

    @classmethod
    def int_k(cls, **kwargs) -> "ModuleQuantConfig":
        default_kwargs = {
            "bit": 8,
            "round_mode": "EVEN",
            "sym_mode": True,
            "narrow_range": False,
            "scale_PoT_round_mode": "CEIL",
            "use_qoperator": False,
            "granularity": "per-tensor",
            "group_channel_size": 1,
            "group_channel_axis": None,
            "signed_mode": True,
            "scale": None,
            "zero_point": None,
            "scale_type": "fp32",
            "scale_int_bit": 16,
            "block_size": 128,
            "group_for_group_conv": None,
            "dynamic_mode": True,
        }
        kwargs = default_kwargs | kwargs
        return cls(
            input=TensorQuantConfig(quantizer="InputINTQuantizer", **kwargs),
            weight=TensorQuantConfig(quantizer="WeightINTQuantizer", **kwargs),
        )

    @classmethod
    def ocp_mxfp4_e2m1(cls, **kwargs):
        default_kwargs = {
            "quantizer": "OCPMXFPQuantizer",
            "bit": 4,
            "block_size": 32,
            "bit_e": 2,
            "granularity": "per-block",
            # applies to mantissa
            "round_mode": "EVEN",
            "bias_mode": "ieee_wo_inf_and_nan",
            "scale_type": "PoT",
            # applies to scale, EVEN if quant_adaptive_share_scale
            "scale_PoT_round_mode": "FLOOR",
        }
        kwargs = default_kwargs | kwargs
        return cls(
            input=TensorQuantConfig(**kwargs),
            weight=TensorQuantConfig(**kwargs),
        )


RoundModes = Literal["EVEN", "AWAY", "CEIL", "INF", "TRUNC", "FLOOR",
                     "STOCHASTIC", "ADAROUND"]
Granularities = Literal["per-tensor", "per-channel", "per-block"]
ScaleTypes = Literal["fp32", "PoT"]
BiasModes = Literal["ieee_wo_inf_and_nan"]
Quantizers = Literal["origin", "INTQuantizer", "WeightINTQuantizer",
                     "InputINTQuantizer", "OCPMXFPQuantizer"]
Observers = Literal["MinMax", "MSE", "Percentile"]


@dataclass
class TensorQuantConfig:
    quantizer: Optional[Quantizers] = None
    bit: Optional[int] = None
    bit_e: Optional[int] = None
    round_mode: Optional[RoundModes] = None
    bias_mode: Optional[BiasModes] = None
    sym_mode: Optional[bool] = None
    narrow_range: Optional[bool] = None
    scale_PoT_round_mode: Optional[RoundModes] = None
    use_qoperator: Optional[bool] = None
    granularity: Optional[Granularities] = None
    group_channel_axis: Optional[int] = None
    group_channel_size: Optional[int] = None
    signed_mode: Optional[bool] = None
    scale: Optional[float | Tensor] = None
    zero_point: Optional[int | float | Tensor] = None
    scale_type: Optional[ScaleTypes] = None
    scale_int_bit: Optional[int] = None
    block_size: Optional[int] = None
    axis: Optional[int] = None
    group_for_group_conv: Optional[int] = None
    dynamic_mode: Optional[bool] = None
    observers: Optional[list[Observers]] = None
    num_calib_samples: Optional[int] = None
    num_calib_mse_bins: int = 256
    calib_percentile_percentage: float = 99.99999

    def get_not_none(self, key: str):
        value = self.__dict__.get(key)
        assert value is not None, f"Expecting config entry: {key}"
        return value

    def __or__(self, other: "TensorQuantConfig"):
        # merging non-None entries from other into self
        for key, value in other.__dict__.items():
            if value is not None:
                if self.__dict__[key] is None:
                    self.__dict__[key] = value
                else:
                    if value != self.__dict__[key]:
                        warnings.warn(
                            f"Conflict value ({self.__dict__[key]} vs. {value}) found for config entry `{key}`. Opt for {self.__dict__[key]}.",
                            UserWarning)
        return self
