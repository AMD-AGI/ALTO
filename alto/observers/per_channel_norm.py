# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT


import torch
from alto.utils.pytorch.module import TransformerConv1D
from .base import Observer, register_observer

PRECISION = torch.float32


@register_observer("per_channel_norm")
class PerChannelNormObserver(Observer):
    """
    A custom observer that computes the L2 norm of each channel and stores it in a buffer.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        self.register_buffer("stats", self.make_empty_row_scalars())
        self.register_buffer("num_samples", torch.zeros([1], dtype=torch.int32, device=self.device))

    def make_empty_row_scalars(self) -> torch.Tensor:
        weight = self.module().weight
        num_columns = weight.shape[1]
        return torch.zeros(num_columns, device=self.device)

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        with torch.no_grad():
            module = self.module()
            inp = x_orig.detach()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)

            num_added = inp.shape[0]  # note this is the number of dataset samples, not
            # multiplied by the sequence length

            if isinstance(module,
                          torch.nn.Linear) or (TransformerConv1D and isinstance(
                              module, TransformerConv1D)):
                if inp.dim() == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()

            if isinstance(module, torch.nn.Conv2d):
                unfold = torch.nn.Unfold(
                    module.kernel_size,
                    dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride,
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)

            self.stats *= self.num_samples / (self.num_samples + num_added)
            self.num_samples += num_added

            inp = inp.type(PRECISION)
            self.stats += torch.norm(inp, p=2, dim=1) ** 2 / self.num_samples

        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.stats.zero_()
            self.num_samples.zero_()

    def calculate_params(self):
        pass
