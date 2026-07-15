# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.protocols.module import Module

class DecomposedLinear(Module):
    
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False
        lora_rank: int = 32
        
    _EXTRA_INIT = {
        "u": partial(nn.init.normal_, std=0.02),
        "v": nn.init.zeros_,
        "sigma": nn.init.ones_,
    }

    def __init__(self, config: Config):
        super().__init__()
        self.in_features = config.in_features
        self.out_features = config.out_features

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if config.bias else None
        self.u = nn.Parameter(torch.empty(config.lora_rank, self.out_features))  # transposed
        self.v = nn.Parameter(torch.empty(self.in_features, config.lora_rank))
        self.sigma = nn.Parameter(torch.empty(config.lora_rank))

    @classmethod
    def from_linear(cls, linear: nn.Linear, lora_rank: int = 32) -> "DecomposedLinear":
        config = cls.Config(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            lora_rank=lora_rank,
        )
        module = cls(config)
        module.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            module.bias.data.copy_(linear.bias.data)
        return module

    def forward(self, input):
        lora_update = (input @ self.v) * self.sigma
        y = F.linear(input, self.weight) + lora_update @ self.u
        if self.bias is not None:
            y += self.bias
        return y
