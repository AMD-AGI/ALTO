# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from .base import Observer, register_observer, MinMaxTuple


def _get_min_max(observed: torch.Tensor) -> MinMaxTuple:
    min_vals = torch.amin(observed, dim=(0, -1))
    max_vals = torch.amax(observed, dim=(0, -1))

    return min_vals, max_vals


@register_observer("minmax")
class MinMaxObserver(Observer):
    """
    Compute quantization parameters by taking the moving average of all min/max values

    :param base_name: str used to name the observer attribute
    :param args: quantization args used to calibrate and quantize the observed value
    :param module: optional module with attached quantization parameters. This argument
        is required to utilize existing qparams such as global_scale or g_idx
    :param **observer_kwargs: keyword arguments for observer initialization
    """

    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)

    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        return _get_min_max(observed)


@register_observer("memoryless_minmax")
class MemorylessMinMaxObserver(MinMaxObserver):

    def __init__(self, *args, **kwargs) -> None:
        kwargs["memoryless"] = True
        super().__init__(*args, **kwargs)
