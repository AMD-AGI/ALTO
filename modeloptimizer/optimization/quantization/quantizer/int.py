import warnings
import torch
from torchtitan.tools.logging import logger

from modeloptimizer.config import TensorQuantConfig, register_quantizers, feature_flags
from modeloptimizer.optimization.quantization.quantizer.base import Quantizer
from modeloptimizer.optimization.quantization.quantizer.utils.round import get_round_func
from modeloptimizer.optimization.quantization.quantizer.utils.shape import pad_shape, de_pad_shape

__all__ = ['INTQuantizer', 'InputINTQuantizer', 'WeightINTQuantizer']


class INTQuantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                input,
                interval,
                zero_point,
                min_q,
                max_q,
                round_func,
                h_alpha=None):
        quant_t = round_func(input / interval) + zero_point
        #quant_t = torch.minimum(torch.maximum(quant_t, min_q), max_q)
        quant_t = torch.clamp(quant_t, min_q, max_q)
        return quant_t

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None, None, None


class INTDeQuantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, interval, zero_point):
        ctx.save_for_backward(interval)
        return (input - zero_point) * interval

    @staticmethod
    def backward(ctx, grad_output):
        interval, zero_point = ctx.saved_tensors
        return (grad_output - zero_point) * interval, None, None


class INTQuantDequantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, interval, zero_point, min_q, max_q, round_func):
        quant_t = round_func(input / interval) + zero_point
        #quant_t = torch.minimum(torch.maximum(quant_t, min_q), max_q)
        quant_t = (torch.clamp(quant_t, min_q, max_q) - zero_point) * interval
        return quant_t

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None, None, None


int_quant_func = INTQuantFunction.apply
int_dequant_func = INTDeQuantFunction.apply
int_quant_dequant_func = INTQuantDequantFunction.apply


@register_quantizers
class INTQuantizer(Quantizer):

    def __init__(self, quant_config: TensorQuantConfig) -> None:
        super().__init__(
            quant_use_qoperator=quant_config.get_not_none("use_qoperator"))

        self.quant_bit = quant_config.get_not_none("bit")
        self.quant_round_mode = quant_config.get_not_none("round_mode")
        self.round_func = get_round_func(self.quant_round_mode)

        self.quant_sym_mode = quant_config.get_not_none("sym_mode")
        self.quant_narrow_range = quant_config.get_not_none("narrow_range")
        self.quant_signed_mode = quant_config.get_not_none("signed_mode")
        self.quant_scale_PoT_round_mode = quant_config.get_not_none(
            "scale_PoT_round_mode")
        self.quant_granularity = quant_config.get_not_none("granularity")
        self.quant_int_tile_size = quant_config.get_not_none("block_size")
        self.quant_int_tile_axis = quant_config.get_not_none("axis")
        self.quant_group_for_group_conv = quant_config.group_for_group_conv
        self.quant_group_channel_size = quant_config.get_not_none(
            "group_channel_size")
        self.quant_group_channel_axis = quant_config.get_not_none(
            "group_channel_axis")
        self.quant_calib_max = None
        self.quant_calib_min = None
        self.quant_scale = quant_config.scale
        self.quant_zero_point = quant_config.zero_point
        self.quant_scale_type = quant_config.get_not_none("scale_type")
        self.quant_scale_quant_int_bit = quant_config.get_not_none(
            "scale_int_bit")
        self.round_func_scale_type_PoT = get_round_func(
            self.quant_scale_PoT_round_mode)
        self.quant_func = int_quant_func
        self.dequant_func = int_dequant_func
        self.quant_dequant_func = int_quant_dequant_func

        if "int" in self.quant_scale_type:
            self.register_buffer('scale_int_power', torch.tensor([0]))
            self.register_buffer('scale_int', torch.tensor([0]))

        self._allow_padding = True

    def extra_repr(self) -> str:
        parent_extra_repr = super().extra_repr()
        if parent_extra_repr:
            parent_extra_repr = f"{parent_extra_repr},\n"
        return f"""{parent_extra_repr}quant_bit={self.quant_bit}, quant_granularity={self.quant_granularity},
quant_zero_point={self.quant_zero_point}, quant_sym_mode={self.quant_sym_mode},
quant_signed_mode={self.quant_signed_mode}, quant_round_mode={self.quant_round_mode},
quant_narrow_range={self.quant_narrow_range}, quant_scale_type={self.quant_scale_type},
quant_scale_quant_int_bit={self.quant_scale_quant_int_bit}, quant_scale_PoT_round_mode={self.quant_scale_PoT_round_mode},
quant_int_tile_size={self.quant_int_tile_size}, quant_int_tile_axis={self.quant_int_tile_axis},
quant_group_channel_size={self.quant_group_channel_size}, quant_group_channel_axis={self.quant_group_channel_axis}"""

    @property
    def is_qoperator_available(self):
        return True

    def before_forward(self, input):
        if feature_flags.AUTO_CAST_SIGNED_INT and not self.quant_signed_mode and self.quant_sym_mode and torch.min(
                input) < 0:
            # scale calculate from current code is different from scale calculate from original code
            # current code: scale = 2*saturation_max / 255, original code: scale = saturation_max/127
            warnings.warn("Automatically converted to SIGNED INT.")
            self.quant_signed_mode = True

        if self.quant_granularity == "per-block":
            self.input_shape = input.shape
            input = pad_shape(input,
                              tile_size=self.quant_int_tile_size,
                              axis=self.quant_int_tile_axis,
                              group=self.quant_group_for_group_conv)
        return input

    def after_forward(self, input):
        if self.quant_granularity == "per-block":
            # if self.quant_int_tile_size < 128:
            #     self.clear_calculated_params()
            input = de_pad_shape(input,
                                 output_shape=self.input_shape,
                                 axis=self.quant_int_tile_axis,
                                 group=self.quant_group_for_group_conv)
        return input

    def calculate_params(self, t):
        self.min_q, self.max_q = self.calculate_quant_max_min()
        saturation_max, saturation_min = self.calculate_saturation(t)

        if self.quant_sym_mode:
            scale, zero_point = self.calculate_symmetric_params(
                saturation_min, saturation_max, self.min_q, self.max_q)
        else:
            scale, zero_point = self.calculate_asymmetric_params(
                saturation_min, saturation_max, self.min_q, self.max_q)

        scale = self.set_scale_type(scale)
        if self.quant_scale is not None:
            self.interval = self.quant_scale
        else:
            self.interval = scale

        if self.quant_zero_point is not None:
            self.zero_point = self.quant_zero_point
        else:
            self.zero_point = zero_point
        self.move_quant_params_to_tensor_device(device=t.device)
        return self.interval, self.max_q, self.min_q, self.zero_point

    def calculate_quant_max_min(self):
        if self.quant_signed_mode:
            if self.quant_narrow_range:
                min_q = -2**(self.quant_bit - 1) + 1
            else:
                min_q = -2**(self.quant_bit - 1)
            max_q = 2**(self.quant_bit - 1) - 1
        else:
            min_q = 0
            max_q = 2**self.quant_bit - 1
        return min_q, max_q

    def calculate_saturation_max_min(self, t):
        if self.quant_granularity == 'per-tensor':
            t_max = torch.max(t)
            t_min = torch.min(t)
        elif self.quant_granularity == 'per-channel':
            assert self.quant_group_channel_axis is not None
            channel_dim = (self.quant_group_channel_axis + len(t.shape)) % len(
                t.shape)
            ori_dim = t.dim()
            if channel_dim != 0:
                t = self.permute_axis_to_first_dim(t, channel_dim)
            # new_shape = t.shape

            t = t.view(t.shape[0], -1)
            t_max = torch.max(t, dim=-1)[0]
            t_min = torch.min(t, dim=-1)[0]

            t_max = t_max.view([t_max.shape[0]] + [1] * (ori_dim - 1))

            if channel_dim != 0:
                t_max = self.unpermute_axis_to_channel_dim(t_max, channel_dim)
                # t = t.view(new_shape)
                # t = self.unpermute_axis_to_channel_dim(t, channel_dim)
            t_min = t_min.view(t_max.shape)
        elif self.quant_granularity == 'per-group-channel':
            assert self.quant_group_channel_axis is not None
            assert self.quant_group_channel_size is not None
            group_dim = (self.quant_group_channel_axis + len(t.shape)) % len(
                t.shape)
            group_size = self.quant_group_channel_size
            ori_dim = t.dim()
            mode = t.shape[group_dim] % group_size
            if mode != 0:
                if not self._allow_padding:
                    raise NotImplementedError(
                        "Unsupported the case that group axis dimension cannot be divided by group_size."
                    )
                logger.info(
                    "group_dim_shape cannot be divided by group_size, it will be padded by zero"
                )
                padding_tensor_shape = list(t.shape)
                # set padding size
                padding_tensor_shape[group_dim] = group_size - mode
                t = torch.cat(
                    [t, torch.zeros(padding_tensor_shape).to(t.device)],
                    dim=group_dim)

            group_count = t.shape[group_dim] // group_size
            split_t = torch.split(t,
                                  split_size_or_sections=group_size,
                                  dim=group_dim)
            concate_t = torch.cat(split_t)
            concat_t_reshape = concate_t.view(group_count, -1)
            per_group_max = torch.max(concat_t_reshape, dim=-1)[0]
            per_group_min = torch.min(concat_t_reshape, dim=-1)[0]
            t_max = torch.repeat_interleave(per_group_max,
                                            repeats=group_size,
                                            dim=0)
            t_min = torch.repeat_interleave(per_group_min,
                                            repeats=group_size,
                                            dim=0)

            if mode != 0:
                t_max = t_max[0:-(group_size - mode)]
                t_min = t_min[0:-(group_size - mode)]

            t_max = t_max.view([t_max.shape[0]] + [1] * (ori_dim - 1))

            if group_dim != 0:
                t_max = self.unpermute_axis_to_channel_dim(t_max, group_dim)

            t_min = t_min.view(t_max.shape)
        elif self.quant_granularity == 'per-block':
            t_max = torch.max(t, dim=t.dim() - 1, keepdim=True)[0]
            t_min = torch.min(t, dim=t.dim() - 1, keepdim=True)[0]
        else:
            raise NotImplementedError(
                "This quant granularity is not supported. Supported Quant Granularity:per-tensor, per-channel, per-group-channel, per-block"
            )

        if self.quant_sym_mode:
            if self.quant_signed_mode:
                saturation_max = torch.max(torch.abs(t_max), torch.abs(t_min))
                saturation_min = -saturation_max
            else:
                if t_min.min() < 0:
                    raise ValueError(
                        f"Found minimal input ({t_min.min()}) < 0 for UINT Symmetric quantization. Please switch to INT Symmetric or UINT Asymmetric."
                    )
                saturation_max = t_max
                saturation_min = torch.zeros_like(t_min,
                                                  dtype=t.dtype,
                                                  device=t.device)
        else:
            # ensure 0 is included in the quantized range
            saturation_max = torch.where(t_max < 0, 0, t_max)
            saturation_min = torch.where(t_min > 0, 0, t_min)
        return saturation_max, saturation_min

    def calculate_saturation(self, t):
        max_val, min_val = self.calculate_saturation_max_min(t)
        return max_val, min_val

    def calculate_asymmetric_params(self,
                                    saturation_min,
                                    saturation_max,
                                    min_q,
                                    max_q,
                                    integral_zero_point=True):
        n = max_q - min_q
        scale = (saturation_max - saturation_min) / n
        scale = torch.where(scale.eq(0), scale + 1, scale)
        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))
        zero_point += min_q
        return scale, zero_point

    def calculate_symmetric_params(self, saturation_min, saturation_max, min_q,
                                   max_q):
        if self.quant_signed_mode or min_q < 0:
            scale = saturation_max / max_q
        else:
            scale = saturation_max / (max_q - min_q)

        scale = torch.where(scale.eq(0), scale + 1, scale)
        zero_point = torch.zeros_like(scale, dtype=scale.dtype)
        return scale, zero_point

    def set_scale_type(self, scale):
        if self.quant_scale_type == 'fp32':
            dequant_scale = scale
        elif self.quant_scale_type == 'PoT':
            dequant_scale = torch.pow(
                2, self.round_func_scale_type_PoT(torch.log2(scale)))
        elif 'int' in self.quant_scale_type:
            # quant fp32 scale to int16: final scale = 2**power * INT16
            _clip_min = -2**(self.quant_scale_quant_int_bit - 1)
            _clip_max = 2**(self.quant_scale_quant_int_bit - 1) - 1
            if 'v2' in self.quant_scale_type:
                mantissa, exponent = torch.frexp(scale)
                scale = mantissa
            _max_val = scale.to(torch.float32)
            #old version: _max_val / _clip_max, it can also be calculate by _max_val / (_clip_max-_clip_min)
            _scale = _max_val / _clip_max

            #old version: torch.ceil
            self.scale_int_power = self.round_func_scale_type_PoT(
                torch.log2(_scale))
            _scale = torch.pow(2, self.scale_int_power)
            quant_int_scale = torch.clamp(self.round_func(scale / _scale),
                                          _clip_min, _clip_max)
            self.scale_int = quant_int_scale
            dequant_scale = self.scale_int * _scale

            if 'v2' in self.quant_scale_type:
                dequant_scale = (2.0**exponent) * dequant_scale
                self.scale_int_power = self.scale_int_power + exponent
        return dequant_scale

    def permute_axis_to_first_dim(self, t, channel_dim):
        ori_dim = t.dim()
        if channel_dim == 1:
            assert ori_dim >= 2
            t = t.permute(1, 0, *range(2, ori_dim)).contiguous()
        elif channel_dim == 2:
            assert ori_dim >= 3
            t = t.permute(channel_dim, 0, 1, *range(3, ori_dim)).contiguous()
        elif channel_dim == 3:
            assert ori_dim >= 4
            t = t.permute(channel_dim, 0, 1, 2, *range(4, ori_dim)).contiguous()
        elif channel_dim == 4:
            assert ori_dim >= 5
            t = t.permute(channel_dim, *range(0, ori_dim - 1)).contiguous()
        else:
            raise NotImplementedError("Don't support >=5-D tensor")
        return t

    def unpermute_axis_to_channel_dim(self, t, channel_dim):
        ori_dim = t.dim()
        if channel_dim == 1:
            assert ori_dim >= 2
            t = t.permute(1, 0, *range(2, ori_dim)).contiguous()
        elif channel_dim == 2:
            assert ori_dim >= 3
            t = t.permute(1, 2, 0, *range(3, ori_dim)).contiguous()
        elif channel_dim == 3:
            assert ori_dim >= 4
            t = t.permute(1, 2, 3, 0, *range(4, ori_dim)).contiguous()
        elif channel_dim == 4:
            assert ori_dim == 5
            t = t.permute(1, 2, 3, 4, 0).contiguous()
        else:
            raise NotImplementedError("Don't support >=5-D tensor")
        return t

    def move_quant_params_to_tensor_device(self, device):
        self.interval = self.interval.to(device)
        self.zero_point = self.zero_point.to(device)

    def get_group_channel_num(self, t):
        assert self.quant_group_channel_axis is not None
        assert self.quant_group_channel_size is not None
        group_dim = (self.quant_group_channel_axis + len(t.shape)) % len(
            t.shape)
        mode = t.shape[group_dim] % self.quant_group_channel_size
        if mode != 0:
            if not self._allow_padding:
                raise NotImplementedError(
                    "Unsupported the case where group axis dimension cannot be divided by group_size."
                )
            logger.info(
                "group_dim_shape cannot be divided by group_size, it will be padded by zero"
            )
            padding_tensor_shape = list(t.shape)
            # set padding size
            padding_tensor_shape[
                group_dim] = self.quant_group_channel_size - mode
            t = torch.cat([t, torch.zeros(padding_tensor_shape).to(t.device)],
                          dim=group_dim)
        group_count = t.shape[group_dim] // self.quant_group_channel_size
        return group_count


@register_quantizers
class InputINTQuantizer(INTQuantizer):

    def __init__(self, quant_config: TensorQuantConfig) -> None:
        super().__init__(quant_config)
        self.quant_dynamic_mode = quant_config.get_not_none("dynamic_mode")
        self._allow_padding = False

    def extra_repr(self) -> str:
        parent_extra_repr = super().extra_repr()
        if parent_extra_repr:
            parent_extra_repr = f"{parent_extra_repr},\n"
        return f"{parent_extra_repr}quant_dynamic_mode={self.quant_dynamic_mode}"

    def calculate_saturation(self, t):
        if self.quant_calib_max is not None and not self.quant_dynamic_mode:
            if self.quant_granularity in ['per-channel', 'per-tensor']:
                max_val = self.quant_calib_max
                min_val = self.quant_calib_min
            else:
                raise NotImplementedError(
                    "Don't support input static per-block quantization")
        else:
            max_val, min_val = self.calculate_saturation_max_min(t)
        return max_val, min_val

    def calculate_saturation_max_min(self, t):
        if self.quant_granularity == 'per-channel':
            logger.info(
                "only support per-token for input. Here per-channel means per-token"
            )
        return super().calculate_saturation_max_min(t)


@register_quantizers
class WeightINTQuantizer(INTQuantizer):

    def __init__(self, quant_config: TensorQuantConfig) -> None:
        super().__init__(quant_config)

    def calculate_saturation_max_min(self, t):
        if self.quant_granularity == 'per-channel':
            if not isinstance(t, torch.nn.parameter.Parameter):
                raise NotImplementedError(
                    'Per channel quantization possible only with '
                    'linear or conv layer weights')
        return super().calculate_saturation_max_min(t)


@register_quantizers
class BiasINTQuantizer(WeightINTQuantizer):
    """BiasINTQuantizer
    
    This quantizer mimics the behavior of the ``quantize_bias_static`` function in the ONNXRuntime quantizer,
    which generates INT32 (or INT16) biases with scales calculated as ``weight_scale * input_scale``.
    
    To quantize biases independently of weights and inputs, use the ``WeightINTQuantizer`` instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        # ref_weight_quantizer = kwargs.pop("ref_weight_quantizer")
        # ref_input_quantizer = kwargs.pop("ref_input_quantizer")
        super().__init__(*args, **kwargs)
        self.ref_weight_quantizer = None
        self.ref_input_quantizer = None

    def calculate_params(self, t):
        assert self.ref_weight_quantizer is not None
        assert self.ref_input_quantizer is not None
        self.min_q, self.max_q = self.calculate_quant_max_min()
        self.interval = self.ref_input_quantizer.interval * self.ref_weight_quantizer.interval
        if torch.any(
                torch.isclose(self.interval,
                              torch.scalar_tensor(0.0, device=t.device))):
            self.interval = torch.ones_like(self.interval, device=t.device)

        self.move_quant_params_to_tensor_device(device=t.device)
        return self.interval, self.max_q, self.min_q, self.zero_point
