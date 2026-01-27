# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/sgpt_base.py
# licensed under the Apache License 2.0

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
    get_prunable_layers,
    match_targets,
)

LAYER_OBSERVER_BASE_NAME = "layer_sparsity"
BLOCK_OBSERVER_BASE_NAME = "block_sparsity"

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
    sparsity: float | list[float] | None
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

    @property
    @abstractmethod
    def observer_name(self) -> Optional[str]:
        return None

    @abstractmethod
    @torch.no_grad()
    def compress_modules(self):
        raise NotImplementedError()

    def on_initialize(self, model: Module, **kwargs) -> bool:
        """
        Initialize and run the SparseGPT algorithm on the current state
        """

        # infer module and sequential targets
        self.sequential_targets = self._infer_sequential_targets(model)
        layers = get_layers(self.sequential_targets, model)
        self._target_layers = get_layers(self.targets,
                                         model)  # layers containing targets

        # infer layer sparsities
        if self.sparsity_profile == "owl":
            logger.info(
                "Using OWL to infer target layer-wise sparsities from "
                f"{len(dataloader) if dataloader else 0} calibration samples..."
            )
            self.sparsity = self._infer_owl_layer_sparsity(
                model, layers, dataloader)

        # get layers and validate sparsity
        if isinstance(self.sparsity, (list, dict)) and len(
                self._target_layers) != len(self.sparsity):
            raise ValueError(
                f"{self.__repr_name__} was initialized with {len(self.sparsity)} "
                f"sparsities values, but model has {len(layers)} target layers")

        self._initialize_observers()
        return True

    def pre_step(self, model: Module, **kwargs):
        self.started_ = True
        for module, observer in self._layer_observers.items():
            observer.enable()

    def post_step(self, model: Module, **kwargs):
        self.compress_modules()
        for module, observer in self._layer_observers.items():
            observer.disable()

    def on_finalize(self, model: Module, **kwargs):
        self.ended_ = True
        self.remove_hooks()
        for module, observer in self._layer_observers.items():
            delattr(module, f"{observer.base_name}_observer")
        self._layer_observers.clear()
        self._module_names.clear()
        self._target_layers.clear()
        self._module_sparsities.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def _initialize_observers(self):
        """
        initialize observers for each layer in the model,
        also initialize the mappings between modules and their names and sparsities
        """

        observer_name = self.observer_name
        base_name = LAYER_OBSERVER_BASE_NAME
        for index, (layer_name,
                    layer) in enumerate(self._target_layers.items()):
            match self.sparsity:
                case dict():
                    layer_sparsity = self.sparsity[layer_name]
                case list():
                    layer_sparsity = self.sparsity[index]
                case _:
                    layer_sparsity = self.sparsity

            for name, module in get_prunable_layers(layer).items():
                name = f"{layer_name}.{name}"

                if match_targets(name, self.ignore)[0]:
                    continue

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

                self._module_names[module] = name
                self._module_sparsities[module] = layer_sparsity

                if observer_name is None:
                    continue

                observer = Observer.create_instance(
                    self.observer_name,
                    base_name=base_name,
                    args=None,
                    module=module,
                    device=device_type,
                )
                module.register_module(f"{base_name}_observer", observer)
                self.register_hook(
                    module,
                    partial(calibrate_input_hook, base_name=base_name),
                    "forward_pre",
                )
                self._layer_observers[module] = observer

    def _infer_sequential_targets(self,
                                  model: torch.nn.Module) -> str | list[str]:
        match self.sequential_targets:
            case None:
                # TODO: support models other than llama3
                return ["TransformerBlock"]
            case str():
                return [self.sequential_targets]
            case _:
                return self.sequential_targets

    # def _infer_owl_layer_sparsity(
    #     self,
    #     model: torch.nn.Module,
    #     layers: dict[str, torch.nn.Module],
    #     dataloader: torch.utils.data.DataLoader,
    # ) -> dict[str, float]:
    #     activations = self._get_activations(model, dataloader)

    #     groups = {}
    #     for name, layer in layers.items():
    #         prunable_layers = get_prunable_layers(layer)
    #         z = [
    #             m.weight.abs() * activations[f"{name}.{n}"].unsqueeze(0)
    #             for n, m in prunable_layers.items()
    #         ]
    #         groups[name] = torch.cat([item.flatten().cpu() for item in z])

    #     del activations

    #     outlier_ratios = {}
    #     for group in groups:
    #         threshold = torch.mean(groups[group]) * self.owl_m
    #         outlier_ratios[group] = (100 *
    #                                  (groups[group] > threshold).sum().item() /
    #                                  groups[group].numel())
    #     outlier_ratios_arr = numpy.array(
    #         [outlier_ratios[k] for k in outlier_ratios])
    #     for k in outlier_ratios:
    #         outlier_ratios[k] = (
    #             outlier_ratios[k] - outlier_ratios_arr.min()) * (
    #                 1 / (outlier_ratios_arr.max() - outlier_ratios_arr.min()) *
    #                 self.owl_lmbda * 2)
    #     outlier_ratios_arr = numpy.array(
    #         [outlier_ratios[k] for k in outlier_ratios])
    #     sparsities = {
    #         k:
    #             1 - (outlier_ratios[k] - numpy.mean(outlier_ratios_arr) +
    #                  (1 - float(self.sparsity))) for k in outlier_ratios
    #     }
    #     logger.info(f"OWL sparsities for sp={self.sparsity} are:")
    #     for k in sparsities:
    #         logger.info(f"Sparsity for {k}: {sparsities[k]}")
    #     return sparsities

    # def _get_activations(self,
    #                      model,
    #                      dataloader,
    #                      nsamples=128) -> dict[str, int]:
    #     from llmcompressor.pipelines.basic import run_calibration

    #     acts = defaultdict(int)

    #     def save_acts(_module, input: tuple[Any, ...] | torch.Tensor,
    #                   name: str):
    #         nonlocal acts
    #         if isinstance(input, tuple):
    #             input = input[0]
    #         acts[name] += 1.0 / nsamples * input.pow(2).sum(dim=(0, 1)).sqrt()

    #     hooks = set(
    #         self.register_hook(mod, partial(save_acts, name=name),
    #                            "forward_pre")
    #         for name, mod in model.named_modules()
    #         if isinstance(mod, torch.nn.Linear) and "lm_head" not in name)
    #     with HooksMixin.disable_hooks(keep=hooks):
    #         run_calibration(model, dataloader)
    #     self.remove_hooks(hooks)

    #     return acts

    def _split_mask_structure(self, mask_structure: str) -> tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
