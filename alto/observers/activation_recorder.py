# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from .base import Observer, register_observer


@register_observer("activation_recorder")
class ActivationRecorder(Observer):
    """
    A custom observer that recoreds the input activations.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        self.activations = []
        self.register_buffer(
            "num_samples",
            torch.zeros([1], dtype=torch.int32, device="cpu"),
        )

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        self.num_samples += 1
        self.activations.append(x_orig.cpu())
        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.activations.clear()
            self.num_samples.zero_()

    def calculate_params(self):
        pass
