import torch
from torchao.quantization import TorchAODType
from .base_quantizer import BaseQuantizer


class IntegerQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.quant_type = 'int-quant'
        self.ao_dtype = TorchAODType(bit)
        if self.sym:
            self.qmin = -(2 ** (self.bit - 1))
            self.qmax = 2 ** (self.bit - 1) - 1
        else:
            self.qmin = 0.0
            self.qmax = 2**self.bit - 1

        self.qmin = torch.tensor(self.qmin)
        self.qmax = torch.tensor(self.qmax)

    def get_tensor_qparams(self, tensor, args={}):
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_tensor_range(tensor, args)
        scales, zeros, qmax, qmin = self.get_qparams(tensor_range, tensor.device)
        return tensor, scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        tensor = torch.clamp(
            self.round_func(tensor / scales.clamp_min(1e-9) + zeros),
            qmin, qmax,
        )
        return tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin, output_scale_factor=1):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales * output_scale_factor, zeros)
        return tensor

    def fake_quant_act_static(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_act = self.reshape_tensor(q_act)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)
        return q_act

    def fake_quant_act_dynamic(self, act, args={}):
        q_act = act
        org_act_shape = q_act.shape
        org_act_dtype = q_act.dtype
        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)
        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)
        return q_act

    def fake_quant_weight_static(self, weight, args):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        output_scale_factor = (
            args['output_scale_factor'] if 'output_scale_factor' in args else 1
        )
        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin, output_scale_factor)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight
        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T
        return q_weight

    def real_quant_weight_static(self, weight, args):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        weight = self.reshape_tensor(weight)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)
        scales = scales * output_scale_factor
        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if self.sym:
            zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)
        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        org_w_shape = weight.shape
        if 'output_scale_factor' in args:
            output_scale_factor = args['output_scale_factor']
            del args['output_scale_factor']
        else:
            output_scale_factor = 1
        weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(weight, args)
        weight = self.quant(weight, scales, zeros, qmax, qmin)
        weight = self.restore_tensor(weight, org_w_shape)
        scales = scales * output_scale_factor
        if self.bit == 8:
            if self.qmin != 0:
                dtype = torch.int8
            else:
                dtype = torch.uint8
        else:
            dtype = torch.int32
        weight = weight.to(dtype)
        if self.sym:
            zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)
        if zeros is not None:
            zeros = zeros.view(qparams_shape)
        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'IntegerQuantizer(bit={self.bit}, sym={self.sym},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )
