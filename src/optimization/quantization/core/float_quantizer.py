class FloatQuantizer(BaseQuantizer):
    def __init__(self, bit, symmetric, granularity, **kwargs):
        super().__init__(bit, symmetric, granularity, **kwargs)
        self.sym = True
        self.quant_type = 'float-quant'
        self.e_bits = int(self.bit[1])
        self.m_bits = int(self.bit[-1])
        self.quant_dtype = DTYPE[(self.e_bits, self.m_bits)]
        self.sign_bits = 1
        self.num_bits = self.quant_dtype.bits
        self.default_bias = 2 ** (self.e_bits - 1)
        self.dst_nbins = 2**self.num_bits

        self.scale_format = self.kwargs.get('scale_format')
        if self.scale_format != "original":
            self.scale_dtype = DTYPE[(int(self.scale_format[1]), int(self.scale_format[-1]))]
        self.do_global_tensor_scaling = self.kwargs.get('do_global_tensor_scaling')
        self.global_tensor_scale = torch.tensor([1])
        
        if 'float_range' in self.kwargs:
            self.qmin, self.qmax = self.kwargs['float_range']
        else:
            self.qmax = torch.tensor(self.quant_dtype.max)
            self.qmin = torch.tensor(self.quant_dtype.min)
    
    def generate_gparam(self, updated_min_val, updated_max_val, dtype=None):
        if dtype is None:
            dtype = updated_min_val.dtype
        min_vals = torch.min(updated_min_val, torch.zeros_like(updated_min_val))
        max_vals = torch.max(updated_max_val, torch.zeros_like(updated_max_val))
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        global_scale = self.scale_dtype.max * self.quant_dtype.max / max_val_pos
        return global_scale.to(dtype).reshape([1])
    
    def get_float_qparams(self, tensor, tensor_range, device):
        min_val, max_val = tensor_range[0], tensor_range[1]
        maxval = torch.max(max_val, -min_val)

        if maxval.dim() == 0 or (maxval.shape[0] != 1 and len(maxval.shape) != len(tensor.shape)):
            maxval = maxval.view([-1] + [1] * (len(tensor.shape) - 1))

        if self.m_bits >= 5:
            maxval = maxval.to(dtype=torch.float32)

        scales = self.global_tensor_scale.to(device) * (maxval / self.qmax.to(device))

        xc = torch.min(torch.max(tensor, -maxval), maxval)
    
        if self.scale_format != "original":
            assert self.scale_format is not None, "Scale format must exist when low_bit_scale enabled!"
            scales = torch.clamp(scales, max=self.scale_dtype.max, min=self.scale_dtype.min)
            scales = self.scale_dtype.cast_to_format(scales)
        return xc, scales

    def get_hqq_qparams(self, tensor, args):
        tensor = tensor.float()
        tensor = self.reshape_tensor(tensor)
        tensor_range = self.get_minmax_range(tensor)
        tensor, scales = self.get_float_qparams(tensor, tensor_range, tensor.device)
        zeros, qmin, qmax = torch.tensor(0), None, None
        best_scales, best_zeros = self.optimize_weights_proximal(
            tensor, scales, zeros, qmax, qmin
        )
        return tensor, best_scales, best_zeros, qmax, qmin

    def get_tensor_qparams(self, tensor, args={}):
        if self.calib_algo == 'hqq':
            return self.get_hqq_qparams(tensor, args)
        else:
            tensor = self.reshape_tensor(tensor)
            tensor_range = self.get_tensor_range(tensor, args)
            tensor, scales = self.get_float_qparams(
                tensor, tensor_range, tensor.device
            )
            zeros, qmin, qmax = torch.tensor(0), self.qmin, self.qmax

            return tensor, scales, zeros, qmax, qmin

    def quant(self, tensor, scales, zeros, qmax, qmin):
        scales[scales == 0] = 1
        scaled_tensor = tensor / scales + zeros
        q_tensor = torch.clip(scaled_tensor, qmin.to(scaled_tensor.device), qmax.to(scaled_tensor.device))
        q_tensor = self.quant_dtype.cast_to_format(q_tensor)
        return q_tensor

    def dequant(self, tensor, scales, zeros):
        tensor = (tensor - zeros) * scales
        return tensor

    def quant_dequant(self, tensor, scales, zeros, qmax, qmin):
        tensor = self.quant(tensor, scales, zeros, qmax, qmin)
        tensor = self.dequant(tensor, scales, zeros)
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

        if self.do_global_tensor_scaling:
            min_val, max_val = torch.aminmax(q_act)
            self.global_tensor_scale = self.generate_gparam(min_val, max_val)

        q_act, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_act, args)

        if self.do_global_tensor_scaling:
            scales = scales.to(self.global_tensor_scale.dtype) / self.global_tensor_scale
        
        q_act = self.quant_dequant(q_act, scales, zeros, qmax, qmin)

        q_act = self.restore_tensor(q_act, org_act_shape).to(org_act_dtype)

        return q_act

    def fake_quant_weight_static(self, weight, args):

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        if 'rounding' in args:
            org_round_func = self.round_func
            self.round_func = lambda x: torch.floor(x) + args['rounding']

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype
        scales, zeros, qmax, qmin = (
            args['scales'],
            args['zeros'],
            args['qmax'],
            args['qmin'],
        )
        q_weight = self.reshape_tensor(q_weight)
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        if 'rounding' in args:
            self.round_func = org_round_func

        return q_weight

    def fake_quant_weight_dynamic(self, weight, args={}):
        if 'dim' in args and 'ic' in args['dim']:
            q_weight = weight.T
        else:
            q_weight = weight

        org_w_shape = q_weight.shape
        org_w_dtype = q_weight.dtype

        if self.do_global_tensor_scaling:
            min_val, max_val = torch.aminmax(q_weight)
            self.global_tensor_scale = self.generate_gparam(min_val, max_val)

        q_weight, scales, zeros, qmax, qmin = self.get_tensor_qparams(q_weight, args)
        
        if self.do_global_tensor_scaling:
            scales = scales.to(self.global_tensor_scale.dtype) / self.global_tensor_scale
        
        q_weight = self.quant_dequant(q_weight, scales, zeros, qmax, qmin)
        q_weight = self.restore_tensor(q_weight, org_w_shape).to(org_w_dtype)

        if 'dim' in args and 'ic' in args['dim']:
            q_weight = q_weight.T

        return q_weight

    def real_quant_weight_static(self, weight, args):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

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

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def real_quant_weight_dynamic(self, weight, args={}):
        assert self.bit in ['e4m3', 'e5m2'], 'Only FP8 E4M3 and E5M2 support real quant'
        dtype = torch.float8_e4m3fn if self.e_bits == 4 else torch.float8_e5m2

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

        weight = weight.to(dtype)
        zeros = None
        if self.granularity == 'per_tensor':
            qparams_shape = 1
        elif self.granularity == 'per_block':
            qparams_shape = (scales.shape[0], scales.shape[2])
        else:
            qparams_shape = (weight.shape[0], -1)

        scales = scales.view(qparams_shape)
        return weight, scales, zeros

    def __repr__(self):
        return (
            f'FloatQuantizer(bit={self.bit},'
            f'e_bits={self.e_bits}, m_bits={self.m_bits},'
            f'granularity={self.granularity},'
            f'kwargs={self.kwargs}, qmin={self.qmin}, qmax={self.qmax})'
        )
