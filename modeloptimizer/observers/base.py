from abc import ABC, abstractmethod
from math import e
from typing import Any, Optional
import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
)
from compressed_tensors.quantization.utils import calculate_qparams, generate_gparam

from weakref import ref
from .helpers import flatten_for_calibration

MinMaxTuple = tuple[torch.Tensor, torch.Tensor]
AVAILABLE_OBSERVERS = {}


def register_observer(name):
    def decorator(cls):
        if name in AVAILABLE_OBSERVERS:
            raise RuntimeError(f"Observer {name} already registered")
        AVAILABLE_OBSERVERS[name] = cls
        return cls
    return decorator


class Observer(ABC, nn.Module):

    def __init__(
        self,
        base_name: str,
        *,
        args: Optional[QuantizationArgs] = None,
        module: Optional[nn.Module] = None,
        device: torch.device = torch.device("cuda"),
        memoryless: bool = False,
        should_calculate_gparam: bool = False,
        should_calculate_qparams: bool = True,
        **observer_kwargs: Any,
    ):
        super().__init__()
        self.base_name = base_name
        self.module = ref(module) if module else None
        self._enabled = True
        self.device = device

        self.args = args
        # populate observer kwargs
        self.args.observer_kwargs = self.args.observer_kwargs or {}
        self.args.observer_kwargs.update(observer_kwargs)

        self.avg_constant = self.args.observer_kwargs.get(
            "averaging_constant", 0.01)
        self.memoryless = memoryless
        self.should_calculate_gparam = should_calculate_gparam
        self.should_calculate_qparams = should_calculate_qparams

        if self.should_calculate_gparam and not self.memoryless:
            self.register_buffer("global_quant_min", torch.tensor([], device=self.device))
            self.register_buffer("global_quant_max", torch.tensor([], device=self.device))
        else:
            self.global_quant_min = None
            self.global_quant_max = None

        if self.should_calculate_qparams and not self.memoryless:
            self.register_buffer("quant_min", torch.tensor([], device=self.device))
            self.register_buffer("quant_max", torch.tensor([], device=self.device))
        else:
            self.quant_min = None
            self.quant_max = None
            
    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False
        self.clear_stats()

    def __repr__(self):
        return f"{self.__class__.__name__}(enabled={self._enabled})"

    @abstractmethod
    def get_current_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate the min and max value of the observed value (without moving average)
        for the purposes of global scale calculation
        """
        raise NotImplementedError()

    def get_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate moving average of min and max values from observed value

        :param observed: value being observed whose shape is
            (num_observations, *qparam_shape, group_size)
        :return: minimum value and maximum value whose shapes are (*qparam_shape, )
        """
        min_vals, max_vals = self.get_current_min_max(observed)
        if self.memoryless:
            self.quant_min = min_vals
            self.quant_max = max_vals
            return min_vals, max_vals

        if self.quant_min.numel() == 0:
            self.quant_min.resize_(min_vals.shape)
            self.quant_min.copy_(min_vals)
        else:
            if self.avg_constant != 1.0:
                # FUTURE: consider scaling by num observations (first dim)
                #         rather than reducing by first dim
                min_vals = self._lerp(self.quant_min, min_vals,
                                      self.avg_constant)
            self.quant_min.copy_(min_vals)

        if self.quant_max.numel() == 0:
            self.quant_max.resize_(max_vals.shape)
            self.quant_max.copy_(max_vals)
        else:
            if self.avg_constant != 1.0:
                max_vals = self._lerp(self.quant_max, max_vals,
                                      self.avg_constant)
            self.quant_max.copy_(max_vals)

        return self.quant_min, self.quant_max

    def get_global_min_max(self, observed: torch.Tensor) -> MinMaxTuple:
        """
        Calculate moving average of min and max values from observed value
        for the purposes of global scale calculation

        :param observed: value being observed whose shape is
            (num_observations, 1, group_size)
        :return: minimum value and maximum value whose shapes are (1, )
        """
        min_vals, max_vals = self.get_current_global_min_max(observed)
        if self.memoryless:
            self.global_quant_min = min_vals
            self.global_quant_max = max_vals
            return min_vals, max_vals

        if self.global_quant_min.numel() == 0:
            self.global_quant_min.resize_(min_vals.shape)
            self.global_quant_min.copy_(min_vals)
        else:
            if self.avg_constant != 1.0:
                min_vals = self._lerp(self.global_quant_min, min_vals,
                                      self.avg_constant)
            self.global_quant_min.copy_(min_vals)

        if self.global_quant_max.numel() == 0:
            self.global_quant_max.resize_(max_vals.shape)
            self.global_quant_max.copy_(max_vals)
        else:
            if self.avg_constant != 1.0:
                max_vals = self._lerp(self.global_quant_max, max_vals,
                                      self.avg_constant)
            self.global_quant_max.copy_(max_vals)

        return self.global_quant_min, self.global_quant_max

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None or (module := self.module()) is None:
            return None

        return getattr(module, f"{self.base_name}_{name}", None)

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return observed

        with torch.no_grad():
            if self.should_calculate_gparam:
                observed_detached = observed.detach().reshape((1, 1, -1))  # per tensor reshape
                global_min_vals, global_max_vals = self.get_global_min_max(observed_detached)

            if self.should_calculate_qparams:
                g_idx = self._get_module_param("g_idx")
                observed_detached = flatten_for_calibration(
                    observed.detach(),
                    self.base_name,
                    self.args,
                    g_idx,
                )
                min_vals, max_vals = self.get_min_max(observed_detached)

        return observed

    def calculate_params(self):
        if self.should_calculate_gparam:
            global_min_vals, global_max_vals = self.global_quant_min, self.global_quant_max
            if self.memoryless:
                self.global_quant_min = None
                self.global_quant_max = None
            global_scale = generate_gparam(global_min_vals, global_max_vals)
        else:
            global_scale = None
            global_min_vals = None
            global_max_vals = None

        if self.should_calculate_qparams:
            min_vals, max_vals = self.quant_min, self.quant_max
            if self.memoryless:
                self.quant_min = None
                self.quant_max = None
            scales, zero_points = calculate_qparams(
                min_vals=min_vals,
                max_vals=max_vals,
                quantization_args=self.args,
                global_scale=global_scale,
            )
        else:
            scales = None
            zero_points = None
            min_vals = None
            max_vals = None
        return scales, zero_points, global_scale

    def clear_stats(self):
        """
        Clear the statistics of the observer after calibration.
        """
        if self.should_calculate_gparam and not self.memoryless:
            self.global_quant_min.resize_(0)
            self.global_quant_max.resize_(0)
        else:
            self.global_quant_min = None
            self.global_quant_max = None
        if self.should_calculate_qparams and not self.memoryless:
            self.quant_min.resize_(0)
            self.quant_max.resize_(0)
        else:
            self.quant_min = None
            self.quant_max = None

    def _lerp(
        self, input: torch.Tensor, end: torch.Tensor, weight: float
    ) -> torch.Tensor:
        """torch lerp_kernel is not implemeneted for all data types"""
        return (input * (1.0 - weight)) + (end * weight)

    @classmethod
    def create_instance(
        cls,
        observer_name: str,
        base_name: str,
        **kwargs: Any,
    ) -> 'Observer':
        if observer_name not in AVAILABLE_OBSERVERS:
            raise RuntimeError(f"Observer {observer_name} not found")
        return AVAILABLE_OBSERVERS[observer_name](base_name, **kwargs)


# class ObserverContainer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.observers = []
#         self._enabled = True

#     def enable(self):
#         self._enabled = True

#     def disable(self):
#         self._enabled = False
#         self.calculate_params()
#         self.clear_stats()

#     def add_observer(self, observer):
#         self.observers.append(observer)

#     def __repr__(self):
#         if not self._enabled:
#             return "ObserverContainer(disabled)"
#         return f"ObserverContainer(\n\t{',\n\t'.join([observer.__repr__() for observer in self.observers])}\n)"

#     def forward(self, x):
#         if not self._enabled:
#             return x
#         for observer in self.observers:
#             x = observer(x)
#         return x

#     def calculate_params(self):
#         results = []
#         with torch.no_grad():
#             for observer in self.observers:
#                 results.append(observer.calculate_params())
#         return results

#     def clear_stats(self):
#         with torch.no_grad():
#             for observer in self.observers:
#                 observer.clear_stats()


# def call_observers(
#     module: nn.Module,
#     base_name: str,
#     value: Optional[torch.Tensor] = None,
# ):
#     """
#     Call a module's attached input/weight/output observer using a provided value.
#     Update the module's scale and zp using the observer's return values.

#     :param module: torch.nn.Module
#     :param base_name: substring used to fetch the observer, scales, and zp
#     :param value: torch.Tensor to be passed to the observer for activations. If
#         param_name is "weight", then the module's weight tensor will be used
#     """

#     if value is None and base_name == "weight":
#         value = module.weight
#     observers: ObserverContainer = getattr(module, f"{base_name}_observers")

#     x = observers(value)
#     return x


# def calibrate_activations(module: nn.Module, value: torch.Tensor,
#                           base_name: str):
#     """
#     Calibrate input or output activations by calling the a module's attached
#     observer.

#     :param module: torch.nn.Module
#     :param base_name: substring used to fetch the observer, scales, and zp
#     :param value: torch.Tensor to be passed to the observer

#     """
#     # If empty tensor, can't update zp/scale
#     # Case for MoEs
#     if value.numel() == 0:
#         return

#     return call_observers(
#         module=module,
#         base_name=base_name,
#         value=value,
#     )


# def calibrate_input_hook(module: nn.Module, args: tuple[torch.Tensor]):
#     """
#     Hook to calibrate input activations.
#     Will call the observers to update the scales/zp before applying
#     input QDQ in the module's forward pass.
#     """
#     assert isinstance(
#         args,
#         tuple) and len(args) == 1, "Input must be a tuple with one element"
#     arg = args[0]
#     assert isinstance(arg, torch.Tensor), "Input must be a tensor"
#     arg = calibrate_activations(module, value=arg, base_name="input")
#     return (arg,)


# def calibrate_output_hook(module: nn.Module, _args: Any,
#                           output: torch.Tensor):
#     """
#     Hook to calibrate output activations.
#     Will call the observers to update the scales/zp before applying
#     output QDQ.
#     """
#     output = calibrate_activations(
#         module,
#         value=output,
#         base_name="output",
#     )
#     # TODO: accumulate errors?
#     # output = forward_quantize(
#     #     module=module,
#     #     value=output,
#     #     base_name="output",
#     #     args=module.quantization_scheme.output_activations,
#     # )
#     return output
