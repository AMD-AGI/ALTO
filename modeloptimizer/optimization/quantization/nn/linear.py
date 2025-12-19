import torch
from torch.nn import Linear
from torch.nn import functional as F
from modeloptimizer.config import ModuleQuantConfig, TensorQuantConfig, register_layer_mapping
from .base import LWeightInputLayer


@register_layer_mapping(Linear)
class LLinear(LWeightInputLayer, Linear):

    def __init__(self, in_features, out_features, bias=True, **kwargs) -> None:
        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         **kwargs)

    def forward_impl(self, input, weight, bias=None):
        if hasattr(self, 'export_mode') and self.export_mode:
            return self.export_handler(
                input, self.qweight,
                self.qbias if self.qbias is not None else self.bias, **{
                    k: getattr(self, k, None)
                    for k in ('stride', 'padding', 'dilation', 'groups')
                })

        output = F.linear(input, weight, bias=bias)
        return output

    @classmethod
    def default_params(cls) -> ModuleQuantConfig:
        return ModuleQuantConfig(
            input=TensorQuantConfig(axis=-1, group_channel_axis=-1),
            weight=TensorQuantConfig(axis=-1, group_channel_axis=0),
        )
