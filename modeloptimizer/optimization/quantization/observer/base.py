from abc import ABC
from typing import Optional
from torchtitan.tools.logging import logger

from modeloptimizer.config import TensorQuantConfig
from modeloptimizer.config.registry import OBSERVERS


class ObserverContainer:

    def __init__(self, quant_config: Optional[TensorQuantConfig] = None):
        """
            observer_mode: list, e.g. [LAObserver, LBObserver,...]
            quant_observer_mode: dict, e.g. {'BFPQuantizer':[QCObserver, ...],...}
            quant_mode: str, e.g. 'BFPQuantizer' default is None.
        """
        self._observers = []
        if quant_config is not None and quant_config.observers is not None:
            for observer in quant_config.observers:
                self._observers += [
                    OBSERVERS[observer](quant_config=quant_config)
                ]

        if len(self._observers) > 1:
            assert self.check_support_multi_observer()

    def extra_repr(self) -> str:
        res = "["
        for observer in self.enabled_observers:
            res += observer.__class__.__name__
        res += "]"
        return res

    def check_support_multi_observer(self):
        is_support = True
        for observer in self.observers:
            is_support = is_support and observer.check_support_multi_observer()
        return is_support

    def before_quant(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.before_quant(t, quantizer)
        return t

    def after_quant(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.after_quant(t, quantizer)
        return t

    def before_dequant(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.before_dequant(t, quantizer)
        return t

    def after_dequant(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.after_dequant(t, quantizer)
        return t

    def before_quantizer(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.before_quantizer(t, quantizer)
        return t

    def after_quantizer(self, t, quantizer):
        for observer in self.enabled_observers:
            t = observer.after_quantizer(t, quantizer)
        return t

    def initial_quantizer(self, quantizer):
        for observer in self.observers:
            observer.initial_quantizer(quantizer)

    def initial_layer(self, layer):
        for observer in self.observers:
            observer.initial_layer(layer)

    def before_layer_forward(self, layer, input: tuple):
        for observer in self.enabled_observers:
            t = observer.before_layer_forward(layer, input)
            if t is not None:
                if not isinstance(t, tuple):
                    t = (t,)
                input = t
        return input

    def after_layer_forward(self, layer, input: tuple, result):
        for observer in self.enabled_observers:
            hook_result = observer.after_layer_forward(layer, input, result)
            if hook_result is not None:
                result = hook_result
        return result

    def add(self, observer):
        if observer is not None:
            self._observers += observer.observers
            if len(self.observers) > 1:
                assert (self.check_support_multi_observer())

    def get_observer_by_observer_class(self, observer_class):
        current_observer = None
        for observer in self.observers:
            if type(observer) == observer_class:
                current_observer = observer
                break
        if current_observer is not None:
            return current_observer
        else:
            logger.error(
                "Can't find observer according to observer_class, Please check your observer_class."
            )

    def enable_observer(self, observer_name):
        observer_class = observer_ops[observer_name]
        not_in_optim_list = True
        for observer in self._observers:
            if isinstance(observer, observer_class):
                observer.enable()
                not_in_optim_list = False
        if not_in_optim_list:
            raise ValueError(
                '{} is not in observer list and cannot be enabled.'.format(
                    observer_name))

    def disable_observer(self, observer_name):
        observer_class = observer_ops[observer_name]
        for observer in self._observers:
            if isinstance(observer, observer_class):
                observer.disable()

    def disable(self):
        for observer in self._observers:
            observer.disable()

    def remove_observer(self, observer_name):
        observer_class = observer_ops[observer_name]
        for i, observer in enumerate(self._observers):
            if isinstance(observer, observer_class):
                removed_observer = self._observers.pop(i)
                return removed_observer

        raise ValueError('{} is not in observer list.'.format(observer_name))

    def remove(self):
        removed_optims = self._observers
        self._observers = []
        return removed_optims

    @property
    def observers(self):
        return self._observers

    @property
    def enabled_observers(self):
        enabled_optims = [optim for optim in self._observers if optim.enabled]
        return enabled_optims


"""Please inheriet from this class for your optimization methods"""


class Observer(ABC):

    def __init__(self) -> None:
        super().__init__()
        self._enabled = True

    """
        in Quantizer
    """

    # before the quant function
    def before_quant(self, t, quantizer):
        return t

    # after the quant function
    def after_quant(self, t, quantizer):
        return t

    # before the dequant function
    def before_dequant(self, t, quantizer):
        return t

    # after the quant function
    def after_dequant(self, t, quantizer):
        return t

    # before the quantizer
    def before_quantizer(self, t, quantizer):
        return t

    #after the quantizer
    def after_quantizer(self, t, quantizer):
        return t

    #when a quantizer is created
    def initial_quantizer(self, quantizer):
        pass

    """
        in LightLayers
    """

    #when a layer is created
    def initial_layer(self, layer):
        pass

    #before layer forward
    def before_layer_forward(self, layer, input: tuple):
        return input

    #after layer forward
    def after_layer_forward(self, layer, input: tuple, result):
        return result

    @staticmethod
    def is_support_multi_observer():
        return True

    @staticmethod
    def check_support_multi_observer():
        return True

    def disable(self):
        self._enabled = False

    def enable(self):
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled
