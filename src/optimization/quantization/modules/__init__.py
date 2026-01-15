import torch.nn as nn
from .fake_quantized_linear import FakeQuantLinear, EffcientFakeQuantLinear
from .real_quantized_linear import VllmRealQuantLinear, SglRealQuantLinear

LINEAR_TYPES = [
    nn.Linear,
    FakeQuantLinear, 
    EffcientFakeQuantLinear,
    VllmRealQuantLinear, 
    SglRealQuantLinear
]
