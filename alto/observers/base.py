# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright contributors to the vLLM project
# Original portions are licensed under the Apache-2.0 License (see upstream vLLM-project licensing).
#
# SPDX-License-Identifier: Apache-2.0 AND MIT

# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/observers/base.py
# modifications:
# - unified memoryless and non-memoryless observers
# - forward pass only observes data statistics (and possibly modifies the data)
# - qparams are calculated after forwarding all data in the step

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
        if self.args is not None:
            self.args.observer_kwargs = self.args.observer_kwargs or {}
            self.args.observer_kwargs.update(observer_kwargs)

            self.avg_constant = self.args.observer_kwargs.get("averaging_constant", 0.01)
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

    def disable(self, clear: bool = True):
        self._enabled = False
        if clear:
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
                min_vals = self._lerp(self.quant_min, min_vals, self.avg_constant)
            self.quant_min.copy_(min_vals)

        if self.quant_max.numel() == 0:
            self.quant_max.resize_(max_vals.shape)
            self.quant_max.copy_(max_vals)
        else:
            if self.avg_constant != 1.0:
                max_vals = self._lerp(self.quant_max, max_vals, self.avg_constant)
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
                min_vals = self._lerp(self.global_quant_min, min_vals, self.avg_constant)
            self.global_quant_min.copy_(min_vals)

        if self.global_quant_max.numel() == 0:
            self.global_quant_max.resize_(max_vals.shape)
            self.global_quant_max.copy_(max_vals)
        else:
            if self.avg_constant != 1.0:
                max_vals = self._lerp(self.global_quant_max, max_vals, self.avg_constant)
            self.global_quant_max.copy_(max_vals)

        return self.global_quant_min, self.global_quant_max

    def _get_module_param(self, name: str) -> Optional[torch.nn.Parameter]:
        if self.module is None or (module := self.module()) is None:
            return None

        return getattr(module, f"{self.base_name}_{name}", None)

    def forward_inner(self, observed: torch.Tensor) -> torch.Tensor:
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

    def forward(self, observed: torch.Tensor) -> torch.Tensor:
        if not self._enabled:
            return observed

        return self.forward_inner(observed)

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

    def _lerp(self, input: torch.Tensor, end: torch.Tensor, weight: float) -> torch.Tensor:
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
