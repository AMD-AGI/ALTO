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

TEACHER_INPUT_OBSERVER_BASE_NAME = "input"
TEACHER_OUTPUT_OBSERVER_BASE_NAME = "output"
STUDENT_INPUT_OBSERVER_BASE_NAME = "student_input"
STUDENT_OUTPUT_OBSERVER_BASE_NAME = "student_output"


def calibrate_input_hook(module: Module, args: Any, base_name: str):
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """
    input_tensor = args[0] if isinstance(args, tuple) else args
    input_tensor = calibrate_activations(module,
                                         value=input_tensor,
                                         base_name=base_name)
    if isinstance(args, tuple):
        args = (input_tensor,) + args[1:]
    else:
        args = input_tensor
    return args

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

    _teacher_input_observers: dict[torch.nn.Module,
                           Observer] = PrivateAttr(default_factory=dict)
    _teacher_output_observers: dict[torch.nn.Module,
                           Observer] = PrivateAttr(default_factory=dict)
    _student_input_observers: dict[torch.nn.Module,
                           Observer] = PrivateAttr(default_factory=dict)
    _student_output_observers: dict[torch.nn.Module,
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
            TEACHER_INPUT_OBSERVER_BASE_NAME,
            self._teacher_input_observers,
        )
        self._initialize_observers(
            self._observer_name,
            TEACHER_OUTPUT_OBSERVER_BASE_NAME,
            self._teacher_output_observers,
        )
        self._initialize_observers(
            self._observer_name,
            STUDENT_INPUT_OBSERVER_BASE_NAME,
            self._student_input_observers,
        )
        self._initialize_observers(
            self._observer_name,
            STUDENT_OUTPUT_OBSERVER_BASE_NAME,
            self._student_output_observers,
        )
        return True

    def pre_step(self, model: Module, **kwargs):
        self.started_ = True
        for module, observer in self._teacher_input_observers.items():
            observer.enable()
        for module, observer in self._teacher_output_observers.items():
            observer.enable()
        for module, observer in self._student_input_observers.items():
            observer.disable()
        for module, observer in self._student_output_observers.items():
            observer.disable()

    def post_step(self, model: Module, **kwargs):


        for module, observer in self._teacher_input_observers.items():
            observer.disable(clear=False)
        for module, observer in self._teacher_output_observers.items():
            observer.disable(clear=False)
        for module, observer in self._student_input_observers.items():
            observer.enable()
        for module, observer in self._student_output_observers.items():
            observer.enable()
            
        print("teacher: ", self._teacher_input_observers["layers.0"].activations)
        print("student:", self._student_input_observers["layers.0"].activations)

        # TODO: implement post-step logic
        raise NotImplementedError("Post-step logic not implemented")


        for module, observer in self._teacher_input_observers.items():
            observer.disable(clear=True)
        for module, observer in self._teacher_output_observers.items():
            observer.disable(clear=True)
        for module, observer in self._student_input_observers.items():
            observer.disable(clear=True)
        for module, observer in self._student_output_observers.items():
            observer.disable(clear=True)


    def on_finalize(self, model: Module, **kwargs):
        self.ended_ = True
        self.remove_hooks()
        for module, observer in self._teacher_observers.items():
            delattr(module, f"{observer.base_name}_observer")
        self._teacher_observers.clear()
        for module, observer in self._student_observers.items():
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
            if base_name.endswith("input"):
                self.register_hook(
                    module,
                    partial(
                        calibrate_input_hook,
                        base_name=base_name,
                    ),
                    "forward_pre",
                )
            elif base_name.endswith("output"):
                self.register_hook(
                    module,
                    partial(
                        calibrate_output_hook,
                        base_name=base_name,
                    ),
                    "forward",
                )
            else:
                raise ValueError(f"Cannot determine where to register the observer: {base_name}")
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
