import math

import numpy as np
import torch
import torch.nn as nn


class VllmRealQuantLinear(nn.Module):
    def __init__(self, weight, bias, scales, input_scale, need_pack, scales_name):
        super().__init__()
        weight_name = 'weight_packed' if need_pack else 'weight'
        self.register_buffer(weight_name, weight)
        self.register_buffer('bias', bias) if bias is not None else setattr(self, 'bias', None)
        self.register_buffer(scales_name, scales)
        self.register_buffer('input_scale', input_scale)

    @torch.no_grad()
    def forward(self, x):
        raise NotImplementedError

    @classmethod
    @torch.no_grad()
    def new(cls, module, w_q, quant_config):
        weight, scales = cls.quant_pack(module, w_q, quant_config)
        if hasattr(module, 'buf_act_scales_0'):
            input_scale = module.buf_act_scales_0
        else:
            input_scale = None
        if (
            'act' in quant_config
            and quant_config.act.get('static', False)
            and quant_config.get('quant_type', 'int-quant') == 'int-quant'
        ):
            input_scale = input_scale.unsqueeze(0)

        if module.bias is not None:
            bias = module.bias.data
        else:
            bias = None

        need_pack = quant_config['weight'].get('need_pack', False)

        if quant_config['weight']['granularity'] == 'per_block':
            scales_name = 'weight_scale_inv'
        else:
            scales_name = 'weight_scale'

        new_module = cls(weight, bias, scales, input_scale, need_pack, scales_name)
        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.weight_shape = weight.shape
        new_module.weight_dtype = weight.dtype
        new_module.scales_shape = scales.shape
        new_module.scales_dtype = scales.dtype

        new_module.zeros_shape = None
        new_module.zeros_dtype = None

        return new_module

    @classmethod
    @torch.no_grad()
    def quant_pack(cls, module, w_q, quant_config):
        weight, scales, zeros = w_q(module)
        need_pack = quant_config['weight'].get('need_pack', False)
        if need_pack:
            weight, scales = cls.pack(weight, scales, quant_config)
        return weight, scales

    @classmethod
    @torch.no_grad()
    def pack(self, weight, scales, quant_config):

        # Packs a tensor of quantized weights stored in int8 into int32s with padding
        num_bits = quant_config['weight']['bit']

        # convert to unsigned for packing
        offset = pow(2, num_bits) // 2
        weight = (weight + offset).to(torch.uint8)
        weight = weight.cpu().numpy().astype(np.uint32)
        pack_factor = 32 // num_bits

        # pad input tensor and initialize packed output
        packed_size = math.ceil(weight.shape[1] / pack_factor)
        packed = np.zeros((weight.shape[0], packed_size), dtype=np.uint32)
        padding = packed.shape[1] * pack_factor - weight.shape[1]
        weight = np.pad(weight, pad_width=[(0, 0), (0, padding)], constant_values=0)

        # pack values
        for i in range(pack_factor):
            packed |= weight[:, i::pack_factor] << num_bits * i

        packed = np.ascontiguousarray(packed).view(np.int32)
        int_weight = torch.from_numpy(packed).cuda()
        del weight, packed
        return int_weight, scales.to(torch.float16)

    def __repr__(self):
        return (
            'VllmRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )


class SglRealQuantLinear(VllmRealQuantLinear):
    def __init__(self, weight, bias, scales, input_scale, need_pack, scales_name):
        super().__init__(weight, bias, scales, input_scale, need_pack, scales_name)

    def __repr__(self):
        return (
            'SglRealQuantLinear('
            + f'in_features={self.in_features}, '
            + f'out_features={self.out_features}, '
            + f'bias={self.bias is not None}, '
            + f'weight_shape={self.weight_shape}, '
            + f'weight_dtype={self.weight_dtype}, '
            + f'scales_shape={self.scales_shape}, '
            + f'scales_dtype={self.scales_dtype}, '
            + f'zeros_shape={self.zeros_shape}, '
            + f'zeros_dtype={self.zeros_dtype})'
        )