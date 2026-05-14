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
        self.u = nn.Parameter(torch.empty(lora_rank, out_features))  # permuted
        self.v = nn.Parameter(torch.empty(in_features, lora_rank))
        self.sigma = nn.Parameter(torch.empty(lora_rank))

    def forward(self, input):
        y = F.linear(input, self.weight) + input @ self.v @ torch.diag(self.sigma) @ self.u
        if self.bias is not None:
            y += self.bias
        return y

    @classmethod
    def from_linear(cls, linear: nn.Linear, lora_rank: int = 32):
        new_layer = cls(linear.in_features, linear.out_features, linear.bias is not None, lora_rank)
        new_layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            new_layer.bias.data.copy_(linear.bias.data)
        return new_layer
