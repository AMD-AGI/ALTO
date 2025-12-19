from typing import Optional
from dataclasses import dataclass

from .quantization_config import QuantConfig, ModuleQuantConfig, TensorQuantConfig

@dataclass
class JobConfig:
    quantization: Optional[QuantConfig] = None
