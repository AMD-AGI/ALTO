from typing import Optional
import torch
from modeloptimizer.config import TensorQuantConfig
from modeloptimizer.config.registry import QUANTIZERS
from modeloptimizer.optimization.quantization.observer.base import ObserverContainer
from modeloptimizer.optimization.quantization.quantizer.base import Quantizer


def get_quantizer(
        quant_config: Optional[TensorQuantConfig]
) -> "HolderQuantizer | Quantizer":
    r"""Generate the specific quantizer using the given configuration, it will return a Quantizer

    """

    if quant_config is None or quant_config.quantizer.lower() == "origin":
        quantizer = HolderQuantizer()
        return quantizer
    quantizer_cls = QUANTIZERS[quant_config.quantizer]
    quantizer = quantizer_cls(quant_config)
    return quantizer


def get_quant_observers(quant_config: Optional[TensorQuantConfig]):
    r"""Generate the specific observers using the given configuration, it will return a Observers

    """
    observer = ObserverContainer(quant_config=quant_config)
    return observer


def QuantizerFactory(
        quant_config: Optional[TensorQuantConfig]) -> "QuantizerSolver":
    r"""Generate the specific quantizer with observers using the given configuration. it will return a QuantizerSolver class

    Args:
        quant_mode: the name of the specific quantizer. if the value is ``origin``, it will return a HolderQuantizer which does not do any observer and Default: ``origin``, 
        quant_observer_mode: it is a dict which the key is quantizer and the name is corresponding observer methods. Default: ``{}``
        **kwargs: the arguments for the specific quantizer

    Examples::

        >>> kwargs = {'quant_bit':8, 'quant_bfp_tile_size':8}
        >>> quantizer = QuantizerFactory(quant_mode='bfpquantizer', quant_observer_mode={'bfpquantizer':'mseobserver'}, **kwargs)

    """
    quantizer = get_quantizer(quant_config)
    observer = get_quant_observers(quant_config)
    quantizer_runner = QuantizerSolver(quantizer, observer)
    return quantizer_runner


class QuantizerSolver(torch.nn.Module):

    def __init__(
        self,
        quantizer: "HolderQuantizer | Quantizer",
        observers: ObserverContainer,
    ) -> None:
        super().__init__()
        self.quantizer = quantizer

        # observer init should be put after the initialization of this Quantizer
        self.observer = observers
        if self.observer:
            self.observer.initial_quantizer(self.quantizer)

    def extra_repr(self) -> Optional[str]:
        if self.observer:
            return f"(observers): {self.observer.extra_repr()}"
        return None

    def forward_observer(self, input, dequantize):
        input = self.observer.before_quantizer(input, self.quantizer)
        #note: self.before_forward is after the observer.before_quantizer
        input = self.quantizer.before_forward(input)
        if self.quantizer.use_qoperator:
            if dequantize:
                input = self.observer.before_dequant(input, self.quantizer)
                if self.quantizer.need_quantization:
                    input = self.quantizer.dequantize(input)
                input = self.observer.after_dequant(input, self.quantizer)
            else:
                input = self.observer.before_quant(input, self.quantizer)
                if self.quantizer.need_calculate_params:
                    _ = self.quantizer.calculate_params(input)
                if self.quantizer.need_quantization:
                    input = self.quantizer.quantize(input)
                input = self.observer.after_quant(input, self.quantizer)
        else:
            #import pdb
            #pdb.set_trace()
            input = self.observer.before_quant(input, self.quantizer)
            if self.quantizer.need_calculate_params:
                self.quantizer.calculate_params(input)
            if self.quantizer.need_quantization:
                input = self.quantizer.quantdequant(input)
            input = self.observer.after_dequant(input, self.quantizer)
        input = self.quantizer.after_forward(input)
        #note: self.after_forward is before the observer.after_quantizer
        input = self.observer.after_quantizer(input, self.quantizer)
        return input

    def forward_wo_observer(self, input, dequantize=False):
        input = self.quantizer.before_forward(input)
        if self.quantizer.use_qoperator:
            if dequantize:
                if self.quantizer.need_quantization:
                    input = self.quantizer.dequantize(input)
            else:
                if self.quantizer.need_calculate_params:
                    _ = self.quantizer.calculate_params(input)
                if self.quantizer.need_quantization:
                    input = self.quantizer.quantize(input)
        else:
            if self.quantizer.need_calculate_params:
                _ = self.quantizer.calculate_params(input)
            if self.quantizer.need_quantization:
                input = self.quantizer.quantdequant(input)
        input = self.quantizer.after_forward(input)
        return input

    def forward(self, input, dequantize=False):
        if self.observer and self.observer.enabled_observers:
            output = self.forward_observer(input, dequantize=dequantize)
        else:
            output = self.forward_wo_observer(input, dequantize)
        return output


class HolderQuantizer():

    def __init__(self, **kwargs) -> None:
        self.observer = None
        self.use_qoperator = False
        self.need_calculate_params = False
        self.need_quantization = False

    def __call__(self, input, dequantize=False):
        return input

    def before_forward(self, input):
        return input

    def after_forward(self, input):
        return input
