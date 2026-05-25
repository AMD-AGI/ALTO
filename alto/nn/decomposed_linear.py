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
        new_layer = cls(linear.in_features, linear.out_features, linear.bias is not None, lora_rank)
        new_layer.weight = linear.weight
        new_layer.bias = linear.bias
        return new_layer

    def init_lora_weights(self, init_std: float = 0.02):
        nn.init.normal_(self.u, mean=0.0, std=init_std)
        nn.init.zeros_(self.v)
        nn.init.ones_(self.sigma)
