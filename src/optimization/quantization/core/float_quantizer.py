
import torch
from torchao.quantization import (
    choose_qparams_affine_floatx, 
    quantize_affine_floatx, 
    dequantize_affine_floatx
)

from .numerical_utils import *
from .data_types import *


class FloatQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.granularity = granularity
        self.kwargs = kwargs
        self.quant_type = 'float-quant'
        self.e_bits, self.m_bits = int(self.bit[1]), int(self.bit[-1])
        self.quant_dtype = DTYPE[(self.e_bits, self.m_bits)]
        
        self.scale_format = self.kwargs.get('scale_format', 'identity')
        if self.scale_format != "identity":
            self.scale_e_bits = int(self.scale_format[1]) 
            self.scale_m_bits = int(self.scale_format[-1])
            self.scale_dtype = DTYPE[(self.scale_e_bits, self.scale_m_bits)]
        self.global_scaling = self.kwargs.get('global_scaling')
        self.global_scale = torch.tensor([1])

    def get_ori_dtype(self, tensor):
        self.ori_dtype = tensor.dtype

    def reshape_tensor(self, tensor):
        reshaped_tensor = tensor
        if self.granularity == 'per_tensor':
            reshaped_tensor = tensor.reshape(1, -1)
        if self.granularity == 'per_channel':
            reshaped_tensor = tensor.reshape(-1, tensor.shape[-1])
        elif self.granularity == 'per_group':
            group_size = self.kwargs['group_size']
            if tensor.shape[-1] % group_size == 0:
                reshaped_tensor = tensor.reshape(-1, group_size)
            else:
                deficiency = group_size - tensor.shape[1] % group_size
                prefix = tensor.shape[:-1]
                pad_zeros = torch.zeros(
                    (*prefix, deficiency), device=tensor.device, dtype=tensor.dtype
                )
                reshaped_tensor= torch.cat((tensor, pad_zeros), dim=-1).reshape(
                    -1, group_size
                )
        elif self.granularity == 'per_token':
            if tensor.ndim == 3:
                num_token = tensor.shape[1]
                reshaped_tensor = tensor.permute(1, 0, 2).reshape(num_token, -1)
            elif tensor.ndim == 4:
                num_token = tensor.shape[2]
                reshaped_tensor = tensor.permute(2, 0, 1, 3).reshape(num_token, -1)
        elif self.granularity == 'per_head':
            assert tensor.ndim == 4
            num_head = tensor.shape[1]
            reshaped_tensor = tensor.permute(1, 0, 2, 3).reshape(num_head, -1)
        return reshaped_tensor

    def restore_tensor(self, tensor, ori_shape):
        restored_tensor = tensor
        if self.granularity == 'per_tensor':
            restored_tensor = tensor.reshape(*ori_shape)
        elif self.granularity == 'per_channel':
            restored_tensor = tensor.reshape(*ori_shape)
        elif self.granularity == 'per_group':
            try:
                restored_tensor = tensor.reshape(ori_shape)
            except:
                group_size = self.kwargs['group_size']
                deficiency = group_size - ori_shape[1] % group_size
                restored_tensor = tensor.reshape(*ori_shape[:-1], -1)[..., :-deficiency]
        elif self.granularity == 'per_token':
            if len(ori_shape) == 3:
                num_token = ori_shape[1]
                restored_tensor = tensor.reshape(num_token, ori_shape[0], ori_shape[2]).permute(1, 0, 2)
            elif len(ori_shape) == 4:
                num_token = ori_shape[2]
                restored_tensor = tensor.reshape(num_token, ori_shape[0], ori_shape[1], ori_shape[3]).permute(1, 2, 0, 3)
        elif self.granularity == 'per_head':
            num_head = ori_shape[1]
            restored_tensor = tensor.reshape(num_head, ori_shape[0], ori_shape[2], ori_shape[3]).permute(1, 0, 2, 3)
        return restored_tensor


    def get_tensor_qparams(self, tensor):
        self.get_ori_dtype(tensor)
        scales = choose_qparams_affine_floatx(
            tensor=tensor,
            ebits=self.e_bits,
            mbits=self.m_bits
        )
        if self.scale_format != "identity":
            if self.global_scaling:
                self.global_scale = tensor.abs().max().to(torch.float32).view(1) / (self.scale_dtype.max * self.quant_dtype.max)
                scaled_scales = scales.to(torch.float32) / self.global_scale
                scaled_scales = self.scale_dtype.cast_scale_to_format(
                    scaled_scales,
                    scaling_mode='floor',
                    emax=self.quant_dtype.exponent_bias
                )
                total_scale = self.global_scale * scaled_scales
                scales = total_scale.to(self.ori_dtype)
            else:
                scales = self.scale_dtype.cast_scale_to_format(
                    scales, 
                    scaling_mode='floor',
                    emax=self.quant_dtype.exponent_bias
                )
        scales[scales == 0] = 1
        return scales

    def quant(self, tensor, scales):
        return quantize_affine_floatx(
            tensor=tensor,
            scale=scales,
            ebits=self.e_bits,
            mbits=self.m_bits
        )

    def dequant(self, tensor, scales):
        return dequantize_affine_floatx(
            tensor=tensor,
            scale=scales,
            ebits=self.e_bits,
            mbits=self.m_bits,
            output_dtype=self.ori_dtype
        )

    def quant_dequant(self, tensor, scales):
        quantized = quantize_affine_floatx(
            tensor=tensor,
            scale=scales,
            ebits=self.e_bits,
            mbits=self.m_bits
        )
        return dequantize_affine_floatx(
            tensor=quantized,
            scale=scales,
            ebits=self.e_bits,
            mbits=self.m_bits,
            output_dtype=self.ori_dtype
        )

    def fake_quant_act_dynamic(self, act):
        q_act = act
        ori_act_shape = q_act.shape
        ori_act_dtype = q_act.dtype
        q_act = self.reshape_tensor(q_act)
        scales = self.get_tensor_qparams(q_act)
        q_act = self.quant_dequant(q_act, scales)
        q_act = self.restore_tensor(q_act, ori_act_shape).to(ori_act_dtype)
        return q_act

    def fake_quant_weight_static(self, weight, args):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        ori_w_shape = q_weight.shape
        ori_w_dtype = q_weight.dtype
        scales = args['scales']
        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(q_weight, scales)
        q_weight = self.restore_tensor(q_weight, ori_w_shape).to(ori_w_dtype)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        ori_w_shape = q_weight.shape
        ori_w_dtype = q_weight.dtype
        q_weight = self.reshape_tensor(q_weight)
        scales = self.get_tensor_qparams(q_weight)
        q_weight = self.quant_dequant(q_weight, scales)
        q_weight = self.restore_tensor(q_weight, ori_w_shape).to(ori_w_dtype)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def real_quant_weight_static(self, weight, args):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2
        ori_w_shape = weight.shape
        scales = args['scales']
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales)
        weight = self.restore_tensor(weight, ori_w_shape)
        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)
        scales = scales.reshape(qparams_shape)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2
        ori_w_shape = weight.shape
        scales = self.get_tensor_qparams(weight)
        weight = self.quant(weight, scales)
        weight = self.restore_tensor(weight, ori_w_shape)
        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)
        scales = scales.reshape(qparams_shape)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'FloatQuantizer(bit={self.bit},'
            f'e_bits={self.e_bits}, m_bits={self.m_bits},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}'
        )
