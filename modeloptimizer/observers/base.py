from abc import ABC, abstractmethod
from typing import Any, Optional
import torch
import torch.nn as nn

AVAILABLE_OBSERVERS = {}


def register_observer(name):
    def decorator(cls):
        if name in AVAILABLE_OBSERVERS:
            raise RuntimeError(f"Observer {name} already registered")
        AVAILABLE_OBSERVERS[name] = cls
        return cls
    return decorator


class ObserverBase(ABC, nn.Module):

    def __init__(
            self,
            device: torch.device = torch.device("cuda"),
    ):
        super().__init__()
        self.device = device

    def forward(self, x):
        return x

    @abstractmethod
    def calculate_params(self):
        pass

    @abstractmethod
    def clear_stats(self):
        """
        Clear the statistics of the observer after calibration.
        """
        pass

    @classmethod
    def from_name(cls, name, **kwargs):
        if name not in AVAILABLE_OBSERVERS:
            raise RuntimeError(f"Observer {name} not found")
        return AVAILABLE_OBSERVERS[name](**kwargs)


class ObserverContainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.observers = []
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False
        self.calculate_params()
        self.clear_stats()

    def add_observer(self, observer):
        self.observers.append(observer)

    def __repr__(self):
        if not self._enabled:
            return "ObserverContainer(disabled)"
        return f"ObserverContainer(\n\t{',\n\t'.join([observer.__repr__() for observer in self.observers])}\n)"

    def forward(self, x):
        if not self._enabled:
            return x
        for observer in self.observers:
            x = observer(x)
        return x

    def calculate_params(self):
        for observer in self.observers:
            observer.calculate_params()

    def clear_stats(self):
        for observer in self.observers:
            observer.clear_stats()


def call_observers(
    module: nn.Module,
    param_name: str,
    value: Optional[torch.Tensor] = None,
):
    """
    Call a module's attached input/weight/output observer using a provided value.
    Update the module's scale and zp using the observer's return values.

    :param module: torch.nn.Module
    :param param_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer for activations. If
        param_name is "weight", then the module's weight tensor will be used
    """

    if value is None and param_name == "weight":
        value = module.weight
    observers: ObserverContainer = getattr(module, f"{param_name}_observers")

    x = observers(value)
    return x


def calibrate_activations(module: nn.Module, value: torch.Tensor,
                          param_name: str):
    """
    Calibrate input or output activations by calling the a module's attached
    observer.

    :param module: torch.nn.Module
    :param param_name: substring used to fetch the observer, scales, and zp
    :param value: torch.Tensor to be passed to the observer

    """
    # If empty tensor, can't update zp/scale
    # Case for MoEs
    if value.numel() == 0:
        return

    return call_observers(
        module=module,
        param_name=param_name,
        value=value,
    )


def calibrate_input_hook(module: nn.Module, args: tuple[torch.Tensor]):
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """
    assert isinstance(
        args,
        tuple) and len(args) == 1, "Input must be a tuple with one element"
    arg = args[0]
    assert isinstance(arg, torch.Tensor), "Input must be a tensor"
    arg = calibrate_activations(module, value=arg, param_name="input")
    return (arg,)


def calibrate_output_hook(module: nn.Module, _args: Any,
                          output: torch.Tensor):
    """
    Hook to calibrate output activations.
    Will call the observers to update the scales/zp before applying
    output QDQ.
    """
    output = calibrate_activations(
        module,
        value=output,
        param_name="output",
    )
    # TODO: accumulate errors?
    # output = forward_quantize(
    #     module=module,
    #     value=output,
    #     param_name="output",
    #     args=module.quantization_scheme.output_activations,
    # )
    return output
