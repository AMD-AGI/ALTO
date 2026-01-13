from typing import Literal
from dataclasses import dataclass

SparsificationMethods = Literal["none", "Wanda", "Magnitude"]
BlockSaliencyMetrics = Literal["absmean", "absmax"]


@dataclass
class SparsificationConfig:
    method: SparsificationMethods = "none"
    sparsity: float = 0.0
    M: int = -1
    N: int = -1
    block_sparsity: bool = False
    block_width: int = 1
    block_height: int = 1
    block_saliency_metric: BlockSaliencyMetrics = "absmean"
    error_accumulation: bool = False
