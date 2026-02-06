from abc import abstractmethod
from functools import partial
from typing import Any, Optional
import gc

import torch
from torch.nn import Module
from pydantic import Field, PrivateAttr, field_validator, model_validator
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type

from modeloptimizer.observers import Observer
from modeloptimizer.modifiers import Modifier
from modeloptimizer.modifiers.quantization.calibration import calibrate_activations
from modeloptimizer.utils.pytorch.module import (
    get_layers,
)

TEACHER_OBSERVER_BASE_NAME = "output"
STUDENT_OBSERVER_BASE_NAME = "student_output"


def calibrate_output_hook(module: Module, _args: Any, output: torch.Tensor, base_name: str):
    """
    Hook to calibrate output activations.
    Will call the observers to update the scales/zp before applying
    output QDQ.
    """
    output = calibrate_activations(
        module,
        value=output,
        base_name=base_name,
    )
    return output


class LayerWiseDistillationModifier(Modifier):

    # modifier arguments

    # data pipeline arguments
    sequential_targets: str | list[str] | None = None
    targets: str | list[str] = ["__all__"]

    # private variables
    _observer_name: str | None = PrivateAttr(default="activation_recorder")
    _module_names: dict[torch.nn.Module,
                        str] = PrivateAttr(default_factory=dict)
    _target_layers: dict[str,
                         torch.nn.Module] = PrivateAttr(default_factory=dict)

    _teacher_observers: dict[torch.nn.Module,
                           Observer] = PrivateAttr(default_factory=dict)
    _student_observers: dict[torch.nn.Module,
                           Observer] = PrivateAttr(default_factory=dict)

    def on_initialize(self, model: Module, **kwargs) -> bool:
        if len(self.targets) == 1 and self.targets[0] == "__all__":
            # infer module and sequential targets
            self.sequential_targets = self._infer_sequential_targets(model)
            self._target_layers = get_layers(self.sequential_targets, model)
        else:
            self._target_layers = get_layers(self.targets, model)

        self._initialize_observers(
            self._observer_name,
            TEACHER_OBSERVER_BASE_NAME,
            self._teacher_observers,
        )
        self._initialize_observers(
            self._observer_name,
            STUDENT_OBSERVER_BASE_NAME,
            self._student_observers,
        )
        return True

    def pre_step(self, model: Module, **kwargs):
        self.started_ = True
        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.enable()
        for name, observer in self._student_observers.items():
            module = self._target_layers[name]
            observer.disable()

    def post_step(self, model: Module, **kwargs):
        input_iterator = kwargs.get("input_iterator")
        output_iterator = kwargs.get("output_iterator")
        metrics_processor = kwargs.get("metrics_processor")
        log_function = kwargs.get("log_function")
        forward_step = kwargs.get("forward_step")

        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.disable(clear=False)

        # TODO: obtain input/output tensors from vLLM actor
        for _microbatch, input_dict in enumerate(input_iterator):
            
            
            for name, observer in self._student_observers.items():
                module = self._target_layers[name]
                observer.enable()
                
            new_result = forward_step(input_dict)
            old_result = next(output_iterator)
            
            # TODO: calculate distillation loss

            print(f"new_result: {new_result}, old_result: {old_result}")
            
            for name, observer in self._student_observers.items():
                module = self._target_layers[name]
                observer.disable(clear=True)

            # log metrics
            if not metrics_processor.should_log(_microbatch):
                continue

            log_function(metrics_processor, _microbatch)

        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.disable(clear=True)


    def on_finalize(self, model: Module, **kwargs):
        self.ended_ = True
        self.remove_hooks()
        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            delattr(module, f"{observer.base_name}_observer")
        self._teacher_observers.clear()
        for name, observer in self._student_observers.items():
            module = self._target_layers[name]
            delattr(module, f"{observer.base_name}_observer")
        self._student_observers.clear()
        self._target_layers.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _initialize_observers(
        self,
        observer_name: str | None,
        base_name: str,
        observers: dict[str, Observer],
    ):
        """
        initialize observers for each target layer in the model
        also initialize the mappings between modules and their names
        """

        if observer_name is None:
            return

        for name, module in self._target_layers.items():
            observer = Observer.create_instance(
                observer_name,
                base_name=base_name,
                args=None,
                module=module,
                device=device_type,
            )
            module.register_module(f"{base_name}_observer", observer)
            self.register_hook(
                module,
                partial(
                    calibrate_output_hook,
                    base_name=base_name,
                ),
                "forward",
            )
            observers[name] = observer

    def _infer_sequential_targets(self,
                                  model: torch.nn.Module) -> str | list[str]:
        match self.sequential_targets:
            case None:
                return ["TransformerBlock"]
            case str():
                return [self.sequential_targets]
            case _:
                return self.sequential_targets
