from dataclasses import dataclass, field
from .quantization_config import QuantConfig, ModuleQuantConfig, TensorQuantConfig
from .sparsification_config import SparsificationConfig


@dataclass
class JobConfig:
    quantization: QuantConfig = field(default_factory=QuantConfig)
    sparsification: SparsificationConfig = field(
        default_factory=SparsificationConfig)
