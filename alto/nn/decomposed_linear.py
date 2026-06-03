# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecomposedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lora_rank=32):
        super(DecomposedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.u = nn.Parameter(torch.empty(lora_rank, out_features))  # transposed
        self.v = nn.Parameter(torch.empty(in_features, lora_rank))
        self.sigma = nn.Parameter(torch.empty(lora_rank))

    def forward(self, input):
        lora_update = (input @ self.v) * self.sigma
        y = F.linear(input, self.weight) + lora_update @ self.u
        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, lora_rank: int = 32):
        # Build u/v/sigma directly on the source weight's device/dtype (which may
        # be "meta" during TorchTitan's meta-device model construction). We must
        # NOT allocate on CPU and then `.to(device=...)`: moving a real CPU tensor
        # onto a meta device triggers an incompatible set_data and crashes during
        # `on_convert`, before `to_empty`/`init_weights` have materialized params.
        device = linear.weight.device
        dtype = linear.weight.dtype

        new_layer = cls.__new__(cls)
        nn.Module.__init__(new_layer)
        new_layer.in_features = linear.in_features
        new_layer.out_features = linear.out_features
        new_layer.weight = linear.weight
        new_layer.bias = linear.bias
        new_layer.u = nn.Parameter(
            torch.empty(lora_rank, linear.out_features, device=device, dtype=dtype)
        )  # transposed
        new_layer.v = nn.Parameter(
            torch.empty(linear.in_features, lora_rank, device=device, dtype=dtype)
        )
        new_layer.sigma = nn.Parameter(
            torch.empty(lora_rank, device=device, dtype=dtype)
        )
        return new_layer
    
    def init_lora_weights(self, init_std: float = 0.02):
        nn.init.normal_(self.u, mean=0.0, std=init_std)
        nn.init.zeros_(self.v)
        nn.init.ones_(self.sigma)
