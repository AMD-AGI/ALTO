from abc import abstractmethod
import torch

__all__ = ['Quantizer']


class BaseQuantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, interval, zero_point, min_q, max_q, round_func):
        quant_t = round_func(input / interval) + zero_point
        # Save context for backward
        # gradient == 1
        grad_mask = torch.le(quant_t, max_q) & torch.ge(quant_t, min_q)
        # gardient == 0.5 at clip boundary, for maximum and minmum
        grad_mask_clip_boundary = (quant_t == max_q) | (quant_t == min_q)

        ctx.save_for_backward(grad_mask, grad_mask_clip_boundary, interval)

        quant_t = torch.minimum(torch.maximum(quant_t, min_q), max_q)
        #quant_t = torch.clamp(quant_t, min_q, max_q)
        return quant_t

    def backward(ctx, grad_output):
        grad_mask, grad_mask_clip_boundary, interval = ctx.saved_tensors

        grad_input = grad_output * grad_mask
        grad_input = grad_input * (1.0 - grad_mask_clip_boundary.to(
            torch.float32) + grad_mask_clip_boundary * 0.5)
        grad_input = grad_input / interval

        return grad_input, None, None, None, None, None


class BaseDeQuantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, interval, zero_point):
        ctx.save_for_backward(interval)
        ctx.save_for_backward(zero_point)
        return (input - zero_point) * interval

    def backward(ctx, grad_output):
        interval = ctx.saved_tensors
        zero_point = ctx.saved_tensors
        return (grad_output - zero_point) * interval, None, None


class BaseQuantDequantFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, interval, zero_point, min_q, max_q, round_func):
        quant_t = round_func(input / interval) + zero_point
        #quant_t = torch.minimum(torch.maximum(quant_t, min_q), max_q)
        grad_mask = torch.le(quant_t, max_q) & torch.ge(quant_t, min_q)
        # gardient == 0.5 at clip boundary, for maximum and minmum
        grad_mask_clip_boundary = (quant_t == max_q) | (quant_t == min_q)
        ctx.save_for_backward(grad_mask, grad_mask_clip_boundary)
        quant_t = (torch.clamp(quant_t, min_q, max_q) - zero_point) * interval
        return quant_t

    @staticmethod
    def backward(ctx, grad_output):
        grad_mask, grad_mask_clip_boundary = ctx.saved_tensors
        grad_input = grad_output * grad_mask
        grad_input = grad_input * (1.0 - grad_mask_clip_boundary.to(
            torch.float32) + grad_mask_clip_boundary * 0.5)
        return grad_output, None, None, None, None, None


base_quant_func = BaseQuantFunction.apply
base_dequant_func = BaseDeQuantFunction.apply
base_quant_dequant_func = BaseQuantDequantFunction.apply


class Quantizer(torch.nn.Module):

    def __init__(self, quant_use_qoperator=False, **kwargs) -> None:
        super().__init__()
        if quant_use_qoperator:
            assert (self.is_qoperator_available)
        self.use_qoperator = quant_use_qoperator
        self.quant_func = base_quant_func
        self.dequant_func = base_dequant_func
        self.quant_dequant_func = base_quant_dequant_func

        self.interval = torch.tensor(0.0)
        self.zero_point = torch.tensor(0.0)
        self.round_func = None
        self.min_q, self.max_q = torch.tensor(0.0), torch.tensor(0.0)

        self.need_calculate_params = True
        self.need_quantization = True
        self._export_mode = False

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if self._export_mode and not value:
            self.clear_export_cache()
        self._export_mode = value

    def clear_export_cache(self):
        pass

    def extra_repr(self) -> str:
        return f"""need_quantization={self.need_quantization}, need_calculate_params={self.need_calculate_params}"""

    def quantize(self, input: torch.Tensor) -> torch.Tensor:
        interval = self.interval
        zero_point = self.zero_point
        round_func = self.round_func
        min_q, max_q = self.min_q, self.max_q
        qinput = self.quant_func(input, interval, zero_point, min_q, max_q,
                                 round_func)
        return qinput

    def dequantize(self, qinput: torch.Tensor) -> torch.Tensor:
        input = self.dequant_func(qinput, self.interval, self.zero_point)
        return input

    def quantdequant(self, input: torch.Tensor) -> torch.Tensor:
        interval = self.interval
        zero_point = self.zero_point
        round_func = self.round_func
        min_q, max_q = self.min_q, self.max_q
        qinput = self.quant_dequant_func(input, interval, zero_point, min_q,
                                         max_q, round_func)
        if qinput.dtype != input.dtype:
            qinput = qinput.to(input.dtype)
        return qinput

    @property
    def is_qoperator_available(self):
        return False

    @abstractmethod
    def calculate_params(self, input):
        return self.interval, self.zero_point, self.max_q, self.min_q

    def before_forward(self, input):
        return input

    def after_forward(self, input):
        return input

    def disable_quantization(self):
        self.need_quantization = False

    def enable_quantization(self):
        self.need_quantization = True

    def disable_calculate_params(self):
        self.need_calculate_params = False

    def enable_calculate_params(self):
        self.need_calculate_params = True

    def clear_calculated_params(self):
        self.interval = torch.tensor(0.0)
        self.min_q, self.max_q = torch.tensor(0.0), torch.tensor(0.0)
