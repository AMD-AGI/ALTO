from dataclasses import dataclass, field
from .quantization_config import QuantConfig, ModuleQuantConfig, TensorQuantConfig


@dataclass
class JobConfig:
    quantization: QuantConfig = field(default_factory=QuantConfig)
