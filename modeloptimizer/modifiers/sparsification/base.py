# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/pruning/sparsegpt/sgpt_base.py
# licensed under the Apache License 2.0

from abc import abstractmethod
from functools import partial
from typing import Any, Generator, Literal
import gc
import re

import torch
from torch.nn import Module
from pydantic import Field, PrivateAttr, model_validator
from torchtitan.models.common.decoder import Decoder
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

DEFAULT_SUBLAYER_GROUPS = [
    ["wq", "wk", "wv", "q_proj", "k_proj", "v_proj"],
    ["wo", "o_proj"],
    ["w1", "w3", "gate_proj", "up_proj"],
    ["w2", "down_proj"],
]

def calibrate_input_hook(module: Module, args: Any, base_name: str):
    args = args[0] if isinstance(args, tuple) else args
    return calibrate_activations(module, value=args, base_name=base_name)


class SparsityModifierBase(Modifier):
    """
    Abstract base class for oneshot sparsity modifiers.

    When the forward pass completes, decoder blocks are processed
    sequentially: for each block, observer statistics are collected from the
    (possibly already-pruned) output of previous blocks, the block's modules
    are compressed, and the block is re-forwarded to produce input for the
    next block.  This matches the original SparseGPT / Wanda papers.
    """

    # modifier arguments
    sparsity: float | list[float] | dict[str, float] | None
    mask_structure: str = "0:0"

    # data pipeline arguments
    sequential_targets: str | list[str] | None = None
    targets: str | list[str] = ["Linear"]
    ignore: list[str] = Field(default_factory=list)
    sequential: bool = True
    sequential_granularity: Literal["block", "sublayer"] = "sublayer"
    sublayer_groups: list[list[str]] = Field(default_factory=lambda: DEFAULT_SUBLAYER_GROUPS)
    num_calibration_samples: int = 128

    # private variables
    _prune_n: int | None = PrivateAttr(default=None)
    _prune_m: int | None = PrivateAttr(default=None)
    _module_names: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _target_layers: dict[str, torch.nn.Module] = PrivateAttr(default_factory=dict)
    _module_sparsities: dict[torch.nn.Module, str] = PrivateAttr(default_factory=dict)
    _layer_observers: dict[torch.nn.Module, Observer] = PrivateAttr(default_factory=dict)
    _observer_name: str | None = PrivateAttr(default=None)

    _model_args: Decoder.Config | None = PrivateAttr(default=None)

    # sequential state
    _sequential_blocks: list[tuple[str, Module]] = PrivateAttr(default_factory=list)
    _block_modules: dict[int, set[Module]] = PrivateAttr(default_factory=dict)
    _captured_inputs: list[tuple[tuple, dict]] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def validate_model_after(model: "SparsityModifierBase") -> "SparsityModifierBase":
        mask_structure = model.mask_structure
        model._prune_n, model._prune_m = model._split_mask_structure(mask_structure)
        return model

    @abstractmethod
    def compress_modules(self):
        raise NotImplementedError()

    def _is_dict_with_digit_keys(self, sparsity: list | dict | float) -> bool:
        if not isinstance(sparsity, dict):
            return False
        return all(isinstance(k, int) for k in sparsity.keys())

    def validate_sparsity(self, model_parts: list[Module]):
        for m in model_parts:
            if isinstance(self.sparsity, (list, dict)):
                if len(self._target_layers) == len(self.sparsity):
                    continue
                if len(self.sparsity) == m.n_layers:
                    continue
                if self._is_dict_with_digit_keys(self.sparsity) and len(self.sparsity) == len(
                        get_layers(self.sequential_targets, m)):
                    continue
                raise ValueError(f"{self.__repr_name__} was initialized with {len(self.sparsity)} "
                                 f"sparsities values, but model has {len(self._target_layers)} target layers")

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        self._model_args = model_parts[0].config
        for m in model_parts:
            self.sequential_targets = self._infer_sequential_targets(m)
            self._target_layers.update(get_layers(self.targets, m))

        self._initialize_module_name_mappings()
        self._initialize_module_sparsities(model_parts)
        self._initialize_observers(
            self._observer_name, LAYER_OBSERVER_BASE_NAME, self._layer_observers,
        )

        # Build sequential block → modules mapping + capture hook
        for m in model_parts:
            block_dict = get_layers(self.sequential_targets, m)
            self._sequential_blocks = list(block_dict.items())
            for blk_idx, (blk_name, block) in enumerate(self._sequential_blocks):
                mods = {mod for mod in block.modules()
                        if mod in self._module_names}
                self._block_modules[blk_idx] = mods
            if self._sequential_blocks:
                self.register_hook(
                    self._sequential_blocks[0][1], self._capture_hook, "forward_pre",
                )

        logger.info(
            f"{self.__class__.__name__}: {len(self._module_names)} modules, "
            f"{len(self._sequential_blocks)} sequential blocks"
        )
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        if self.sequential:
            for observer in self._layer_observers.values():
                observer.disable()
            self._captured_inputs.clear()
        else:
            for observer in self._layer_observers.values():
                observer.enable()
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        if not self.sequential:
            logger.info(f"{self.__class__.__name__}.on_post_step: mode=global-single-forward")
            return self._fallback_post_step()
        if not self._captured_inputs:
            logger.warning("No block inputs captured; falling back to global-single-forward")
            return self._fallback_post_step()

        mode = f"sequential-{self.sequential_granularity}"
        logger.info(
            f"{self.__class__.__name__}.on_post_step: mode={mode}, "
            f"{len(self._captured_inputs)} microbatches, "
            f"{len(self._sequential_blocks)} blocks"
        )
        block_inputs = self._captured_inputs
        n_blocks = len(self._sequential_blocks)

        for blk_idx, (blk_name, block) in enumerate(self._sequential_blocks):
            block_mods = self._block_modules.get(blk_idx, set())

            if not block_mods:
                logger.info(f"  Block {blk_idx}/{n_blocks} [{blk_name}]: no target modules, propagating")
                block_inputs = self._forward_block(block, block_inputs)
                continue

            if self.sequential_granularity == "sublayer":
                self._process_block_sublayer(blk_idx, blk_name, block, block_inputs, block_mods)
            else:
                self._process_block_whole(blk_idx, blk_name, block, block_inputs, block_mods)

            logger.info(f"  Block {blk_idx}/{n_blocks} [{blk_name}]: block-level re-forward")
            block_inputs = self._forward_block(block, block_inputs)

        self._cleanup()
        return True

    def _process_block_whole(self, blk_idx, blk_name, block, block_inputs, block_mods):
        """Collect stats + compress for all modules in a block at once."""
        n_blocks = len(self._sequential_blocks)
        for mod in block_mods:
            if mod in self._layer_observers:
                self._layer_observers[mod].clear_stats()
                self._layer_observers[mod].enable()
        logger.info(
            f"  Block {blk_idx}/{n_blocks} [{blk_name}]: "
            f"collecting stats, {len(block_mods)} modules"
        )
        with torch.no_grad():
            for inp_args, inp_kwargs in block_inputs:
                block(*inp_args, **inp_kwargs)
        for mod in block_mods:
            if mod in self._layer_observers:
                self._layer_observers[mod].disable(clear=False)
        logger.info(f"  Block {blk_idx}/{n_blocks} [{blk_name}]: compressing")
        self._scoped_compress(block_mods)

    def _process_block_sublayer(self, blk_idx, blk_name, block, block_inputs, block_mods):
        """Split block_mods into sub-groups (qkv/o/up-gate/down) and process
        each with a fresh block forward so later groups see compressed earlier ones."""
        n_blocks = len(self._sequential_blocks)
        mod_name_map = {}
        for mod in block_mods:
            for name, m in block.named_modules():
                if m is mod:
                    mod_name_map[mod] = name
                    break

        remaining = set(block_mods)
        for grp_idx, patterns in enumerate(self.sublayer_groups):
            group_mods = {
                mod for mod in remaining
                if any(p in mod_name_map.get(mod, "") for p in patterns)
            }
            if not group_mods:
                continue
            remaining -= group_mods
            logger.info(
                f"  Block {blk_idx}/{n_blocks} [{blk_name}] sub-group {grp_idx}: "
                f"{[mod_name_map.get(m, '?') for m in group_mods]}"
            )
            self._process_block_whole(blk_idx, blk_name, block, block_inputs, group_mods)

        if remaining:
            logger.info(
                f"  Block {blk_idx}/{n_blocks} [{blk_name}] remaining: "
                f"{[mod_name_map.get(m, '?') for m in remaining]}"
            )
            self._process_block_whole(blk_idx, blk_name, block, block_inputs, remaining)

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        self.ended_ = True
        self._cleanup()
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True

    # ---- sequential helpers -------------------------------------------

    def _capture_hook(self, module: Module, args: Any, kwargs: Any = None):
        if len(self._captured_inputs) >= self.num_calibration_samples:
            return
        if kwargs is None:
            kwargs = {}
        if isinstance(args, torch.Tensor):
            args = (args,)
        self._captured_inputs.append((args, kwargs))

    @staticmethod
    def _forward_block(block: Module, block_inputs):
        new_inputs = []
        with torch.no_grad():
            for inp_args, inp_kwargs in block_inputs:
                out = block(*inp_args, **inp_kwargs)
                if isinstance(out, tuple):
                    new_inputs.append((out, inp_kwargs))
                else:
                    new_inputs.append(((out,) + inp_args[1:], inp_kwargs))
        return new_inputs

    def _scoped_compress(self, block_mods: set[Module]):
        """Call compress_modules with _layer_observers / _module_names /
        _module_sparsities scoped to *block_mods* only."""
        saved = (self._layer_observers, self._module_names, self._module_sparsities)
        self._layer_observers = {m: o for m, o in saved[0].items() if m in block_mods}
        self._module_names = {m: n for m, n in saved[1].items() if m in block_mods}
        self._module_sparsities = {m: s for m, s in saved[2].items() if m in block_mods}
        try:
            with torch.no_grad():
                self.compress_modules()
        finally:
            self._layer_observers, self._module_names, self._module_sparsities = saved

    def _fallback_post_step(self):
        """Non-sequential: enable all observers retroactively (stats from full-model forward)."""
        with torch.no_grad():
            self.compress_modules()
        for observer in self._layer_observers.values():
            observer.disable()
        self._cleanup()
        return True

    def _cleanup(self):
        self.remove_hooks()
        for module, observer in self._layer_observers.items():
            attr = f"{observer.base_name}_observer"
            if hasattr(module, attr):
                delattr(module, attr)
        self._layer_observers.clear()
        self._module_names.clear()
        self._target_layers.clear()
        self._module_sparsities.clear()
        self._sequential_blocks.clear()
        self._block_modules.clear()
        self._captured_inputs.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # ---- module / sparsity helpers ------------------------------------

    def _target_layer_iterator(self) -> Generator[int | None, str, Module]:
        for layer_name, layer in self._target_layers.items():
            for name, module in get_prunable_layers(layer).items():
                name = f"{layer_name}{'.' if name else ''}{name}"
                if match_targets(name, self.ignore)[0]:
                    continue
                index_candidates = re.search(r'\.\d+\.', name)
                if index_candidates is not None:
                    index = int(index_candidates.group(0).strip("."))
                else:
                    index = None
                if isinstance(module, torch.nn.Embedding):
                    continue
                if name.endswith("output"):
                    logger.warning("`output` was previously auto-ignored by SparseGPT and Wanda "
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
        self, observer_name: str | None, base_name: str,
        observers: dict[str, Observer],
    ):
        if observer_name is None:
            return
        for module, name in self._module_names.items():
            observer = Observer.create_instance(
                observer_name, base_name=base_name,
                args=None, module=module, device=device_type,
            )
            object.__setattr__(module, f"{base_name}_observer", observer)
            self.register_hook(
                module,
                partial(calibrate_input_hook, base_name=base_name),
                "forward_pre",
            )
            observers[module] = observer

    def _split_mask_structure(self, mask_structure: str) -> tuple[int, int]:
        n, m = mask_structure.split(":")
        return int(n), int(m)
