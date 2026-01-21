# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, TYPE_CHECKING

import torch
import torch.nn as nn

from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
)

from torchtitan.components.quantization import (
    QuantizationConverter,)
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.utils import device_type
from torchtitan.tools.logging import logger

from modeloptimizer.observers import (
    ObserverBase,
    ObserverContainer,
    calibrate_input_hook,
    calibrate_output_hook,
)

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

SUPPORTED_MODULES = (nn.Linear)
SUPPORTED_PARAM_KEYS = ("input", "weight", "bias", "output")


def _initialize_observer(
    module: nn.Module,
    module_config: dict,
    module_name: str,
):
    observers = dict[str, ObserverContainer]()
    for param_name, param_config in module_config.items():
        assert param_name in SUPPORTED_PARAM_KEYS, f"Unsupported param key: {param_name}"
        observer_container = ObserverContainer()
        for observer_name in param_config["observers"]:
            observer = ObserverBase.from_name(
                observer_name,
                param_name,
                args=QuantizationArgs(num_bits=8),
                module=module,
                device=device_type,
            )
            observer_container.add_observer(observer)
        observer_container_name = f"{param_name}_observers"
        module.register_module(observer_container_name, observer_container)
        observers[f"{module_name}.{observer_container_name}"] = observer_container

    return observers


def _initialize_hooks(module: torch.nn.Module,) -> Set["RemovableHandle"]:
    hooks = set()
    for k in SUPPORTED_PARAM_KEYS:
        optimizer_container = getattr(module, f"{k}_observers", None)
        if optimizer_container is None:
            continue

        match k:
            case "input":
                hooks.add(
                    module.register_forward_pre_hook(calibrate_input_hook))
            case "output":
                hooks.add(module.register_forward_hook(calibrate_output_hook))
            case _:
                raise ValueError(f"Unsupported param key: {k}")

    return hooks


class ModelOptConverter(QuantizationConverter):

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        self._resolved_config = dict()
        self._observer_hooks = set()
        self._observer_container_collection = dict[str, ObserverContainer]()
        self.job_config = job_config

    def resolve_config(self, model: nn.Module, job_config: JobConfig):
        # TODO: resolve from config
        for name, module in model.named_modules(prefix=""):
            if isinstance(module, SUPPORTED_MODULES):
                observers = ["PerChannelNorm", "MinMax"]
                self._resolved_config[name] = {
                    "input": {
                        "observers": observers,
                    }
                }

    def initialize_observers(self, model: torch.nn.Module):
        """
        Attach observers, register activation calibration hooks

        :param model: model to prepare for calibration
        """
        for module_name, module_config in self._resolved_config.items():
            module = model.get_submodule(module_name)
            observers = _initialize_observer(module, module_config, module_name)
            self._observer_container_collection.update(observers)
            self._observer_hooks |= _initialize_hooks(module)

    def enable_observers(self):
        for observer_container in self._observer_container_collection.values():
            observer_container.enable()

    def disable_observers(self):
        for observer_container in self._observer_container_collection.values():
            observer_container.disable()

    def convert(self, model: nn.Module):
        self.resolve_config(model, self.job_config)

    def pre_step(self, model: nn.Module | list[nn.Module]):
        self.enable_observers()

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        for fqn, observer_container in self._observer_container_collection.items():
            qparams = observer_container.calculate_params()
            print(f"{fqn} qparams: {qparams}")

        self.disable_observers()
        raise NotImplementedError("Post step Not implemented")

    def post_initialization(self, model: nn.Module):
        self.initialize_observers(model)
        logger.info(f"Model part after initialization: {model}")


register_model_converter(ModelOptConverter, "modeloptimizer")
