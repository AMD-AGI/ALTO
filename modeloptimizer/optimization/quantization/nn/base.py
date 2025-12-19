from typing import Optional
from abc import abstractmethod, ABC
from modeloptimizer.config import ModuleQuantConfig, TensorQuantConfig
from modeloptimizer.optimization.quantization.observer.base import ObserverContainer
from modeloptimizer.optimization.quantization.quantizer.factory import QuantizerFactory
from modeloptimizer.optimization.quantization.quantizer.int import BiasINTQuantizer
import torch

__all__ = [
    'LCommonLayer', 'LWeightInputLayer'
]


class LCommonLayer(torch.nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: support observers independent of quantizers
        self.observer = ObserverContainer()

        self._export_mode = False
        self.export_handler = None
        self.requires_export_handler = True

    @property
    def export_mode(self):
        return self._export_mode

    @export_mode.setter
    def export_mode(self, value):
        if value and self.training:
            raise RuntimeError(
                "Can't enter export mode during training, only during inference"
            )
        if value and self.requires_export_handler and self.export_handler is None:
            raise RuntimeError(
                "Can't enable export mode on a layer without an export handler")
        elif value and not self.requires_export_handler and self.export_handler is None:
            return  # don't set export mode when it's not required and there is no handler
        elif value and not self._export_mode and self.export_handler is not None:
            self.export_handler.prepare_for_export(self)
            self.export_handler.attach_debug_info(self)
        elif not value and self.export_handler is not None:
            self.export_handler = None
        self._export_mode = value

    def create_quantizer(self, quant_config: Optional[TensorQuantConfig]):
        quantizer = QuantizerFactory(quant_config)
        self.observer.add(quantizer.observer)
        return quantizer

    def forward(self, *inputs):
        if hasattr(self, 'export_mode') and self.export_mode:
            return self._inner_forward(*inputs)

        result = self.observer.before_layer_forward(self, inputs)
        if result is not None:
            if not isinstance(result, tuple):
                result = (result,)
            inputs = result
        result = self._inner_forward(*inputs)
        hook_result = self.observer.after_layer_forward(self, inputs, result)
        if hook_result is not None:
            result = hook_result
        return result

    @abstractmethod
    def _inner_forward(self, *inputs):
        r"""Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    @classmethod
    def default_params(cls) -> ModuleQuantConfig:
        return ModuleQuantConfig()

    def adjust_quant_config(self, quant_config: ModuleQuantConfig):
        return quant_config


class LWeightInputLayer(LCommonLayer, ABC):

    def __init__(self, *args, **kwargs) -> None:
        quant_config: ModuleQuantConfig = kwargs.pop("quant_config")
        super().__init__(*args, **kwargs)
        assert hasattr(self, 'weight'), f'{type(self)} does not contain weights'

        quant_config = self.adjust_quant_config(quant_config)

        self.input_quantizer = self.create_quantizer(quant_config.input)
        self.weight_quantizer = self.create_quantizer(quant_config.weight)
        self.output_quantizer = self.create_quantizer(quant_config.output)
        self.bias_quantizer = self.create_quantizer(quant_config.bias)

        self.observer.initial_layer(self)

        if isinstance(self.bias_quantizer, BiasINTQuantizer):
            self.bias_quantizer.quantizer.ref_input_quantizer = self.input_quantizer.quantizer
            self.bias_quantizer.quantizer.ref_weight_quantizer = self.weight_quantizer.quantizer

        self.qweight = None
        self.qbias = None

    @abstractmethod
    def forward_impl(self, input, weight, bias=None):
        pass

    def _inner_forward(self, input):
        if hasattr(self, 'export_mode') and self.export_mode:
            return self.forward_impl(input, self.weight, bias=self.bias)

        input = self.input_quantizer(input)
        weight = self.weight_quantizer(self.weight)
        bias = self.bias
        if self.bias is not None:
            bias = self.bias_quantizer(self.bias)

        if self.weight_quantizer.quantizer.use_qoperator and self.input_quantizer.quantizer.use_qoperator:
            # TODO: real kernels
            raise NotImplementedError
        else:
            output = self.forward_impl(input, weight, bias)
            output = self.output_quantizer(output)

        return output
