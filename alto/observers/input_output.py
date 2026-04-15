# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
from alto.utils.pytorch.module import TransformerConv1D
from .base import Observer, register_observer

PRECISION = torch.bfloat16


@register_observer("input_output")
class InputOutputObserver(Observer):
    """
    Input-output observer: collects the raw layer inputs and outputs during calibration.
    Each forward input is concatenated along dim=0 (batch), e.g. single input
    [1, 32, 2048] -> after N calls stats is [N_total, 32, 2048]. No reshape/transpose.
    Each forward output is concatenated along dim=0 (batch), e.g. single output
    [1, 32, 2048] -> after N calls stats is [N_total, 32, 2048]. No reshape/transpose.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        self.register_buffer("num_samples", torch.zeros([1], dtype=torch.int32, device=self.device))
        # Placeholder with numel()==0 so first forward assigns; shape unknown until first input
        self.register_buffer("input_stats", torch.zeros(0, dtype=PRECISION, device=self.device))
        self.register_buffer("output_stats", torch.zeros(0, dtype=PRECISION, device=self.device))

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        inp = x_orig.detach().to(PRECISION)
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # [B, C] -> [1, B, C] so batch is always dim=0
        num_added = inp.shape[0]
        self.num_samples += num_added
        if self.stats.numel() == 0:
            self.input_stats = inp
        else:
            self.input_stats = torch.cat([self.input_stats, inp], dim=0)

        out = self.module()(inp)
        if out.dim() == 2:
            out = out.unsqueeze(0)  # [B, C] -> [1, B, C] so batch is always dim=0
        if self.output_stats.numel() == 0:
            self.output_stats = out
        else:
            self.output_stats = torch.cat([self.output_stats, out], dim=0)
        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.num_samples.zero_()
            self.input_stats = torch.zeros(0, dtype=PRECISION, device=self.device)
            self.output_stats = torch.zeros(0, dtype=PRECISION, device=self.device)

    def calculate_params(self):
        pass
