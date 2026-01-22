import torch
from torchao.quantization import (
    MappingType, 
    choose_qparams_affine, 
    fake_quantize_affine, 
    quantize_affine, 
    dequantize_affine
)


class IntegerQuantizer(object):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        self.bit = bit
        self.sym = symmetric
        self.granularity = granularity
        self.quant_type = 'int-quant'
        self.kwargs = kwargs
        self.ao_real_dtype = torch.int32
        self.ao_abstract_dtype = torch.int32
        if self.sym:
            self.ao_sym = MappingType.SYMMETRIC
            self.qmin = -(2 ** (self.bit - 1))
            self.qmax = 2 ** (self.bit - 1) - 1
            if self.bit == 8:
                self.ao_real_dtype = torch.int8
                self.ao_abstract_dtype = torch.int8
        else:
            self.ao_sym = MappingType.ASYMMETRIC
            self.qmin = 0.0
            self.qmax = 2**self.bit - 1
            if self.bit == 8:
                self.ao_real_dtype = torch.uint8
                self.ao_abstract_dtype = torch.uint8

    def get_ori_dtype(self, tensor):
        self.ori_dtype = tensor.dtype

    def get_block_size(self, tensor, granularity):
        block_size = list(tensor.shape)       # per-tensor
        if granularity == 'per_channel':
            block_size[0] = 1
        elif granularity == 'per_group':
            block_size[:-1] = [1] * (len(block_size) - 1)
            block_size[-1] = self.kwargs.get('group_size', block_size[-1])
        elif granularity == 'per_token':
            block_size[-2] = 1
        elif granularity == 'per_block':
            block_size[:-2] = [1] * (len(block_size) - 2)
            block_size[-1] = self.kwargs.get('block_size', block_size[-1])
            block_size[-2] = self.kwargs.get('block_size', block_size[-2])
        elif granularity == 'per_head':
            block_size[-3] = 1
        return torch.Size(block_size)

    def get_tensor_qparams(self, tensor, args={}):
        self.get_ori_dtype(tensor)
        block_size = self.get_block_size(tensor, self.granularity)
        qparams = choose_qparams_affine(
            input=tensor,
            mapping_type=self.ao_sym,
            block_size=block_size,
            target_dtype=self.ao_abstract_dtype,
            quant_min=self.qmin,
            quant_max=self.qmax
        )
        scales, zeros = qparams[0], qparams[1]
        return scales, zeros

    def quant(self, tensor, scales, zeros):
        block_size = self.get_block_size(tensor, self.granularity)
        return quantize_affine(
            input=tensor,
            block_size=block_size,
            scale=scales,
            zero_point=zeros,
            output_dtype=self.ao_real_dtype,
            quant_min=self.qmin,
            quant_max=self.qmax
        )

    def dequant(self, tensor, scales, zeros):
        block_size = self.get_block_size(tensor, self.granularity)
        return dequantize_affine(
            input=tensor,
            block_size=block_size,
            scale=scales,
            zero_point=zeros,
            input_dtype=self.ao_real_dtype,
            quant_min=self.qmin,
            quant_max=self.qmax,
            output_dtype=self.ori_dtype
        )

    def quant_dequant(self, tensor, scales, zeros):
        block_size = self.get_block_size(tensor, self.granularity)
        return fake_quantize_affine(
            input=tensor,
            block_size=block_size,
            scale=scales,
            zero_point=zeros,
            quant_dtype=self.ao_abstract_dtype,
            quant_min=self.qmin,
            quant_max=self.qmax,
        )

    def fake_quant_act_static(self, act, args={}):
        q_act = act
        scales, zeros = (args['scales'], args['zeros'])
        q_act = self.quant_dequant(q_act, scales, zeros)
        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        q_act = act
        scales, zeros = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros)
        return q_act

    def fake_quant_weight_static(self, weight, args):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        scales, zeros = (args['scales'], args['zeros'])
        q_weight = self.quant_dequant(q_weight, scales, zeros)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        scales, zeros = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def real_quant_weight_static(self, weight, args):
        scales, zeros = (args['scales'], args['zeros'])
        weight = self.quant(weight, scales, zeros)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        scales, zeros = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'IntegerQuantizer(bit={self.bit}, sym={self.sym},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )
