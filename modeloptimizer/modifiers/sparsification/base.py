# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/sgpt_base.py
# licensed under the Apache License 2.0

from abc import abstractmethod
from functools import partial
from typing import Any, Optional, Generator
import gc
import re

import numpy
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
    get_prunable_layers,
    match_targets,
)

LAYER_OBSERVER_BASE_NAME = "sparsity"
OWL_OBSERVER_BASE_NAME = "sparsity_owl"

def calibrate_input_hook(module: Module, args: Any, base_name: str):
    """
    Hook to calibrate input activations.
    Will call the observers to update the scales/zp before applying
    input QDQ in the module's forward pass.
    """
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name=base_name)

class SparsityModifierBase(Modifier):
    """
    Abstract base class which implements functionality related to oneshot sparsity.
    Inheriters must implement `calibrate_module` and `compress_modules`
    """

    # modifier arguments
    sparsity: float | list[float] | dict[str, float] | None
    sparsity_profile: str | None = None
    mask_structure: str = "0:0"
    owl_m: int | None = None
    owl_lmbda: float | None = None

    # data pipeline arguments
    sequential_targets: str | list[str] | None = None
    targets: str | list[str] = ["Linear"]
    ignore: list[str] = Field(default_factory=list)

    # private variables
    _prune_n: int | None = PrivateAttr(default=None)
    _prune_m: int | None = PrivateAttr(default=None)
    _module_names: dict[torch.nn.Module,
                        str] = PrivateAttr(default_factory=dict)
    _target_layers: dict[str,
                         torch.nn.Module] = PrivateAttr(default_factory=dict)
    _module_sparsities: dict[torch.nn.Module,
                             str] = PrivateAttr(default_factory=dict)
    _layer_observers: dict[torch.nn.Module,
                      Observer] = PrivateAttr(default_factory=dict)
    _observer_name: str | None = PrivateAttr(default=None)
    _owl_observer_name: str | None = PrivateAttr(default=None)
    _owl_layer_observers: dict[torch.nn.Module, Observer] = PrivateAttr(default_factory=dict)
    _owl_target_sparsity: float | None = PrivateAttr(default=None)

    @field_validator("sparsity_profile", mode="before")
    def validate_sparsity_profile(cls, value: str | None) -> bool:
        if value is None:
            return value

        value = value.lower()

        profile_options = ["owl"]
        if value not in profile_options:
            raise ValueError(f"Please choose profile from {profile_options}")

        return value

    @model_validator(mode="after")
    def validate_model_after(
            model: "SparsityModifierBase") -> "SparsityModifierBase":
        profile = model.sparsity_profile
        owl_m = model.owl_m
        owl_lmbda = model.owl_lmbda
        mask_structure = model.mask_structure

        has_owl_m = owl_m is not None
        has_owl_lmbda = owl_lmbda is not None
        has_owl = profile == "owl"
        owl_args = (has_owl_m, has_owl_lmbda, has_owl)
        if any(owl_args) and not all(owl_args):
            raise ValueError(
                'Must provide all of `profile="owl"`, `owl_m` and `owl_lmbda` or none'
            )

        model._prune_n, model._prune_m = model._split_mask_structure(
            mask_structure)

        return model

    @abstractmethod
    def compress_modules(self):
        raise NotImplementedError()

    def _is_dict_with_digit_keys(self, sparsity: list | dict | float) -> bool:
        if not isinstance(sparsity, dict):
            return False
        return all(isinstance(k, int) for k in sparsity.keys())

    def validate_sparsity(self, model_parts: list[Module]):
        """
        Validate that the sparsity is properly configured
        """
        for m in model_parts:
            if isinstance(self.sparsity, (list, dict)):
                # matches the length of target layers
                if len(self._target_layers) == len(self.sparsity):
                    continue
                # matches the total number of layers in model definition
                if len(self.sparsity) == m.n_layers:
                    continue
                # matches the number of blocks in current model part
                if self._is_dict_with_digit_keys(self.sparsity) and len(
                        self.sparsity) == len(
                            get_layers(self.sequential_targets, m)):
                    continue

                raise ValueError(
                    f"{self.__repr_name__} was initialized with {len(self.sparsity)} "
                    f"sparsities values, but model has {len(self._target_layers)} target layers"
                )

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        """
        Initialize and run the SparseGPT algorithm on the current state
        """

        # infer module and sequential targets
        for m in model_parts:
            self.sequential_targets = self._infer_sequential_targets(m)
            self._target_layers.update(get_layers(
                self.targets, m))  # layers containing targets

        self._initialize_module_name_mappings()

        # infer layer sparsities
        if self.sparsity_profile == "owl":
            logger.info("Using OWL to infer target layer-wise sparsities")
            assert isinstance(self.sparsity, float), "OWL requires a single sparsity value"
            self._owl_target_sparsity = self.sparsity
            self.sparsity = {}
            if self._observer_name != "per_channel_norm":
                self._owl_observer_name = "per_channel_norm"
                # initialize observers for OWL
                self._initialize_observers(
                    self._owl_observer_name,
                    OWL_OBSERVER_BASE_NAME,
                    self._owl_layer_observers,
                )
            else:
                self._owl_layer_observers = self._layer_observers
        else:
            self._initialize_module_sparsities(model_parts)

        self._initialize_observers(
            self._observer_name,
            LAYER_OBSERVER_BASE_NAME,
            self._layer_observers,
        )
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        for module, observer in self._layer_observers.items():
            observer.enable()
        for module, observer in self._owl_layer_observers.items():
            observer.enable()
            
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        if self.sparsity_profile == "owl":
            for m in model_parts:
                blocks = get_layers(self.sequential_targets, m)
                self.sparsity.update(self._infer_owl_layer_sparsity(blocks))
            self._initialize_module_sparsities(model_parts)
        with torch.no_grad():
            self.compress_modules()
        for module, observer in self._layer_observers.items():
            observer.disable()
        for module, observer in self._owl_layer_observers.items():
            observer.disable()

        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        self.ended_ = True
        self.remove_hooks()
        for module, observer in self._layer_observers.items():
            delattr(module, f"{observer.base_name}_observer")
        self._layer_observers.clear()
        for module, observer in self._owl_layer_observers.items():
            delattr(module, f"{observer.base_name}_observer")
        self._owl_layer_observers.clear()
        self._module_names.clear()
        self._target_layers.clear()
        self._module_sparsities.clear()
        gc.collect()
        torch.cuda.empty_cache()

        return True

    def _target_layer_iterator(self) -> Generator[int | None, str, Module]:
        for layer_name, layer in self._target_layers.items():
            for name, module in get_prunable_layers(layer).items():
                name = f"{layer_name}{'.' if name else ''}{name}"

                if match_targets(name, self.ignore)[0]:
                    continue

                # find layer index, which is the first number between two dots
                index_candidates = re.search(r'\.\d+\.', name)
                if index_candidates is not None:
                    index = int(index_candidates.group(0).strip("."))
                else:
                    index = None

                # HACK: previously, embeddings were not quantized because they were not
                # accessible by the layer compressor. For now, we manually ignore it,
                # but in the FUTURE this should be ignored by the user
                if isinstance(module, torch.nn.Embedding):
                    continue

                if name.endswith("output"):
                    logger.warning(
                        "`output` was previously auto-ignored by SparseGPT and Wanda "
                        "modifiers and is not advised. Please add `re:.*output` to "
                        "your ignore list if this was unintentional")

                yield index, name, module

    def _initialize_module_name_mappings(self):
        for index, name, module in self._target_layer_iterator():
            self._module_names[module] = name

    @torch.no_grad()
    def _initialize_module_sparsities(self, model_parts: list[Module]):
        self.validate_sparsity(model_parts)
        self._module_sparsities.clear()
        for index, name, module in self._target_layer_iterator():
            match self.sparsity:
                case dict():
                    if self._is_dict_with_digit_keys(self.sparsity):
                        layer_sparsity = self.sparsity[index]
                    else:
                        layer_sparsity = self.sparsity[name]
                case list():
                    assert index is not None, f"layer index is not found for {name}"
                    layer_sparsity = self.sparsity[index]
                case _:
                    layer_sparsity = self.sparsity
            self._module_sparsities[module] = layer_sparsity

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

        for module, name in self._module_names.items():
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
                    calibrate_input_hook,
                    base_name=base_name,
                ),
                "forward_pre",
            )
            observers[module] = observer

    def _infer_sequential_targets(self,
                                  model: torch.nn.Module) -> str | list[str]:
        match self.sequential_targets:
            case None:
                return ["TransformerBlock"]
            case str():
                return [self.sequential_targets]
            case _:
                return self.sequential_targets

    @torch.no_grad()
    def _infer_owl_layer_sparsity(
        self,
        layers: dict[str, torch.nn.Module],
    ) -> dict[str, float]:
        groups = {}
        for name, layer in layers.items():
            prunable_layers = get_prunable_layers(layer)
            z = [
                m.weight.abs() * self._owl_layer_observers[m].stats.unsqueeze(0)
                for n, m in prunable_layers.items()
            ]
            groups[name] = torch.cat([item.flatten().cpu() for item in z])

        outlier_ratios = {}
        for group in groups:
            threshold = torch.mean(groups[group]) * self.owl_m
            outlier_ratios[group] = (100 *
                                     (groups[group] > threshold).sum().item() /
                                     groups[group].numel())
        outlier_ratios_arr = numpy.array(
            [outlier_ratios[k] for k in outlier_ratios])
        for k in outlier_ratios:
            outlier_ratios[k] = (
                outlier_ratios[k] - outlier_ratios_arr.min()) * (
                    1 / (outlier_ratios_arr.max() - outlier_ratios_arr.min()) *
                    self.owl_lmbda * 2)
        outlier_ratios_arr = numpy.array(
            [outlier_ratios[k] for k in outlier_ratios])
        sparsities = {
            int(k.split(".")[-1]):
                1 - (outlier_ratios[k] - numpy.mean(outlier_ratios_arr) +
                     (1 - float(self._owl_target_sparsity))) for k in outlier_ratios
        }
        logger.info(f"OWL sparsities for sp={self._owl_target_sparsity} are:")
        for k in sparsities:
            logger.info(f"Sparsity for layer #{k}: {sparsities[k]}")
        return sparsities

    def _split_mask_structure(self, mask_structure: str) -> tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
