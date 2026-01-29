import torch
from torchtitan.models.llama3 import (
    Llama3StateDictAdapter as OriginalLlama3StateDictAdapter,)
from modeloptimizer.components import StateDictAdapterMixin

__all__ = ["Llama3StateDictAdapter"]


class Llama3StateDictAdapter(OriginalLlama3StateDictAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add quantization states
        StateDictAdapterMixin.populate_extra_map(self)

    def _permute(self, w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return w

    def _reverse_permute(self, w: torch.Tensor, *args,
                         **kwargs) -> torch.Tensor:
        return w
