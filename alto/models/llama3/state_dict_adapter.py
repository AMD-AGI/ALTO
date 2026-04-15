# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from torchtitan.models.llama3 import (
    Llama3StateDictAdapter as OriginalLlama3StateDictAdapter,)
from alto.components import StateDictAdapterMixin

__all__ = ["Llama3StateDictAdapter"]


class Llama3StateDictAdapter(OriginalLlama3StateDictAdapter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add quantization states
        StateDictAdapterMixin.populate_extra_map(self)

    def map_ignore_list_to_hf(self, ignore_list: list[str]) -> list[str]:
        return StateDictAdapterMixin.map_ignore_list_to_hf(self, ignore_list)
    
    def update_storage_plan(self, state_dict: dict[str, torch.Tensor]):
        StateDictAdapterMixin.update_storage_plan(self, state_dict)

    def _permute(self, w: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return w

    def _reverse_permute(self, w: torch.Tensor, *args,
                         **kwargs) -> torch.Tensor:
        return w
