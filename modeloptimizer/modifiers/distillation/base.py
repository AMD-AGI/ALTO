from abc import abstractmethod
from functools import partial
from typing import Any, Literal
import gc

import torch
from torch.nn import Module
from torch.nn.modules.loss import _Loss as Loss
from pydantic import Field, PrivateAttr, field_validator, model_validator
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_type
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.components.lr_scheduler import LRSchedulersContainer

from modeloptimizer.observers import Observer
from modeloptimizer.modifiers import Modifier
from modeloptimizer.modifiers.quantization.calibration import calibrate_activations
from modeloptimizer.utils.pytorch.module import (
    get_layers,
)
from modeloptimizer.modifiers.distillation.utils import losses

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


class SelfDistillationModifier(Modifier):

    # modifier arguments
    criterion: str | dict[str, str] = "LogitsDistillationLoss"
    loss_weights: dict[str, float] | None = None
    # optimizer arguments
    optimizer: Literal["Adam", "AdamW"] = "AdamW"
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    implementation: Literal["for-loop", "foreach", "fused"] = "fused"
    # lr scheduler arguments
    warmup_steps: int = 0
    decay_ratio: float | None = None
    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    min_lr_factor: float = 0.0
    steps: int = 10

    # data pipeline arguments
    sequential_targets: str | list[str] | None = None
    targets: str | list[str] = ["__all__"]

    # private variables
    _observer_name: str | None = PrivateAttr(default="activation_recorder")
    _module_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _target_layers: dict[str, torch.nn.Module] = PrivateAttr(default_factory=dict)

    _teacher_observers: dict[torch.nn.Module, Observer] = PrivateAttr(default_factory=dict)
    _student_observers: dict[torch.nn.Module, Observer] = PrivateAttr(default_factory=dict)
    _target_loss_fns: dict[str, Loss] = PrivateAttr(default_factory=dict)

    _optimizers: OptimizersContainer | None = PrivateAttr(default=None)
    _lr_schedulers: LRSchedulersContainer | None = PrivateAttr(default=None)

    @property
    def requires_replay_buffer(self) -> bool:
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            if len(self.targets) == 1 and self.targets[0] == "__all__":
                # infer module and sequential targets
                self.sequential_targets = (self._infer_sequential_targets(m))
                self._target_layers.update(get_layers(self.sequential_targets, m))
            else:
                self._target_layers.update(get_layers(self.targets, m))

        for name in self._target_layers.keys():
            self._target_loss_fns[name] = self._get_loss_fn(name)
        self._target_loss_fns["output"] = self._get_loss_fn("output")

        self._build_optimizers(model_parts)
        lr_scheduler_config = LRSchedulersContainer.Config(
            warmup_steps=self.warmup_steps,
            total_steps=None,
            decay_ratio=self.decay_ratio,
            decay_type=self.decay_type,
            min_lr_factor=self.min_lr_factor,
        )
        self._lr_schedulers = lr_scheduler_config.build(
            optimizers=self._optimizers,
            training_steps=self.steps,
        )

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

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        self.started_ = True
        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.enable()
        for name, observer in self._student_observers.items():
            module = self._target_layers[name]
            observer.disable()

        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        is_last_step = kwargs.get("is_last_step")
        # if is_last_step:
        #     return True

        input_iterator = kwargs.get("input_iterator")
        output_iterator = kwargs.get("output_iterator")
        metrics_processor = kwargs.get("metrics_processor")
        log_function = kwargs.get("log_function")
        forward_step = kwargs.get("forward_step")

        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.disable(clear=False)

        local_valid_tokens = torch.tensor(0, dtype=torch.int64)

        for _microbatch, batch in enumerate(input_iterator):
            input_dict, labels = batch
            # TODO: support gradient accumulation once we moved the high precision agent into vLLM
            self._optimizers.zero_grad()
            lr = self._lr_schedulers.schedulers[0].get_last_lr()[0]
            # TODO: change to += if gradient accumulation is supported
            local_valid_tokens = (labels != IGNORE_INDEX).sum().to(device_type)

            for name, observer in self._student_observers.items():
                module = self._target_layers[name]
                observer.enable()

            student_result = forward_step({
                k: v.to(device_type) for k, v in input_dict.items()
            }, labels, local_valid_tokens)
            teacher_result = next(output_iterator).to(device_type)

            loss_values = {}
            loss_values["output"] = self._target_loss_fns["output"](student_result, teacher_result)

            for name, observer in self._student_observers.items():
                module = self._target_layers[name]
                student_activation = observer.activations[0].to(device_type)
                teacher_activation = getattr(
                    module, f"{TEACHER_OBSERVER_BASE_NAME}_observer").activations[_microbatch].to(device_type)
                loss_values[name] = self._target_loss_fns[name](student_activation, teacher_activation)

            # balance the losses
            if self.loss_weights is not None:
                assert len(self.loss_weights) == len(
                    loss_values), "Number of loss weights does not correspond to number of loss values"
                assert sum(self.loss_weights.values()) == 1.0, "Loss weights do not sum to 1.0"
                loss_values = {k: v * self.loss_weights[k] for k, v in loss_values.items()}
                aggregate_loss = sum(loss_values.values())
            else:
                aggregate_loss = sum(loss_values.values()) / len(loss_values)

            aggregate_loss.backward()
            self._optimizers.step()
            self._lr_schedulers.step()

            for name, observer in self._student_observers.items():
                module = self._target_layers[name]
                observer.disable(clear=True)

            # log metrics
            if not metrics_processor.should_log(_microbatch):
                continue

            log_function(
                metrics_processor,
                _microbatch,
                student_loss=loss_values["output"],
                aggregate_loss=aggregate_loss,
                lr=lr,
            )

        for name, observer in self._teacher_observers.items():
            module = self._target_layers[name]
            observer.disable(clear=True)

        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
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

        return True

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

    def _get_loss_fn(self, name: str):
        if isinstance(self.criterion, str):
            return getattr(losses, self.criterion)()
        elif isinstance(self.criterion, dict):
            return getattr(losses, self.criterion[name])()
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")

    def _build_optimizers(self, model_parts: list[Module]):
        config = OptimizersContainer.Config(
            name=self.optimizer,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            implementation=self.implementation,
        )

        self._optimizers = config.build(model_parts=model_parts,)

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True
