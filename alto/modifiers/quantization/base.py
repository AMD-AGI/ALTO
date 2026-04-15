# Quantization modifier base with optional sequential (layer-by-layer) processing.
#
# RTN:  sequential=False (default) — one full-model forward, then update_weight_zp_scale.
# GPTQ: sequential=True  — per-block Hessian collection → GPTQ quantize → re-forward.
# AWQ:  sequential=True  — per-block activation stats → scale search → apply → re-forward.
#
# Subclasses override _process_block() for algorithm-specific per-block logic.

import gc
from typing import Any, Literal, Optional, Union

import torch
import tqdm
from compressed_tensors.quantization import disable_quantization, enable_quantization
from compressed_tensors.utils import getattr_chain, match_named_modules
from pydantic import Field, PrivateAttr
from torch.nn import Module
from torchtitan.tools.logging import logger

from alto.modifiers import Modifier
from alto.modifiers.quantization.mixin import QuantizationMixin
from alto.modifiers.quantization.calibration import update_weight_zp_scale
from alto.utils.pytorch.module import get_layers

__all__ = ["QuantizationModifier"]

DEFAULT_SUBLAYER_GROUPS = [
    ["wq", "wk", "wv", "q_proj", "k_proj", "v_proj"],
    ["wo", "o_proj"],
    ["w1", "w3", "gate_proj", "up_proj"],
    ["w2", "down_proj"],
]


class QuantizationModifier(Modifier, QuantizationMixin):
    """Base quantization modifier (RTN).

    When ``sequential=True``, decoder blocks are processed one at a time via
    :meth:`_process_block`.  Subclasses (GPTQ, AWQ) override that method for
    algorithm-specific per-block logic.

    :param sequential: process blocks one-by-one (default ``False`` for RTN)
    :param sequential_targets: decoder block class name(s); ``None`` auto-infers
    :param sequential_granularity: ``"block"`` processes all modules in a block
        at once; ``"sublayer"`` splits into sub-groups (qkv / o / up-gate / down)
    :param sublayer_groups: list of name-pattern lists defining sub-groups;
        defaults to ``[["wq","wk","wv"], ["wo"], ["w1","w3"], ["w2"]]``
    :param num_calibration_samples: max microbatches captured for sequential mode
    """

    sequential: bool = False
    sequential_targets: Optional[Union[str, list[str]]] = None
    sequential_granularity: Literal["block", "sublayer"] = "sublayer"
    sublayer_groups: list[list[str]] = Field(default_factory=lambda: DEFAULT_SUBLAYER_GROUPS)
    num_calibration_samples: int = 128

    # sequential state (shared by all subclasses)
    _sequential_blocks: list[tuple[str, Module]] = PrivateAttr(default_factory=list)
    _block_modules: dict[int, set[Module]] = PrivateAttr(default_factory=dict)
    _captured_inputs: list[tuple[tuple, dict]] = PrivateAttr(default_factory=list)

    # ---- lifecycle ----------------------------------------------------

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        if not QuantizationMixin.has_config(self):
            raise ValueError("QuantizationModifier requires that quantization fields be specified")
        for m in model_parts:
            QuantizationMixin.initialize_quantization(self, m)

        if self.sequential:
            self._build_sequential_blocks(model_parts)
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            QuantizationMixin.start_calibration(self, m)
        if self.sequential:
            self._captured_inputs.clear()
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            QuantizationMixin.end_calibration(self, m)

        if self.sequential and self._captured_inputs:
            mode = f"sequential-{self.sequential_granularity}"
            logger.info(
                f"{self.__class__.__name__}.on_post_step: mode={mode}, "
                f"{len(self._captured_inputs)} microbatches, "
                f"{len(self._sequential_blocks)} blocks"
            )
            return self._sequential_post_step(model_parts)
        logger.info(f"{self.__class__.__name__}.on_post_step: mode=global-single-forward")
        return self._nonsequential_post_step(model_parts)

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            QuantizationMixin.clear_calibration(self, m)
        self._sequential_blocks.clear()
        self._block_modules.clear()
        self._captured_inputs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True

    # ---- sequential loop (template) -----------------------------------

    def _sequential_post_step(self, model_parts: list[Module]) -> bool:
        """Process blocks one at a time.  Calls :meth:`_process_block` then
        re-forwards to propagate quantized outputs to the next block."""
        block_inputs = self._captured_inputs
        n_blocks = len(self._sequential_blocks)

        for blk_idx, (blk_name, block) in enumerate(self._sequential_blocks):
            block_mods = self._block_modules.get(blk_idx, set())

            if not block_mods:
                logger.info(f"  Block {blk_idx}/{n_blocks} [{blk_name}]: no modules, propagating")
                block.apply(disable_quantization)
                block_inputs = self._forward_block(block, block_inputs)
                block.apply(enable_quantization)
                continue

            if self.sequential_granularity == "sublayer":
                self._process_block_sublayer(
                    blk_idx, blk_name, block, block_inputs, block_mods,
                )
            else:
                logger.info(
                    f"  Block {blk_idx}/{n_blocks} [{blk_name}]: "
                    f"{len(block_mods)} modules, {len(block_inputs)} microbatches"
                )
                self._process_block(blk_idx, blk_name, block, block_inputs, block_mods)

            logger.info(f"  Block {blk_idx}/{n_blocks} [{blk_name}]: block-level re-forward")
            block.apply(disable_quantization)
            block_inputs = self._forward_block(block, block_inputs)
            block.apply(enable_quantization)

        self._post_sequential_cleanup(model_parts)
        return True

    def _process_block_sublayer(self, blk_idx, blk_name, block, block_inputs, block_mods):
        """Split *block_mods* into sub-groups and process each with a fresh
        block forward, so later sub-groups see compressed earlier sub-groups."""
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

            group_names = [mod_name_map.get(m, "?") for m in group_mods]
            logger.info(
                f"  Block {blk_idx}/{n_blocks} [{blk_name}] sub-group {grp_idx}: "
                f"{group_names}"
            )
            self._process_block(blk_idx, blk_name, block, block_inputs, group_mods)

        if remaining:
            logger.info(
                f"  Block {blk_idx}/{n_blocks} [{blk_name}] remaining: "
                f"{[mod_name_map.get(m, '?') for m in remaining]}"
            )
            self._process_block(blk_idx, blk_name, block, block_inputs, remaining)

    def _nonsequential_post_step(self, model_parts: list[Module]) -> bool:
        """All-at-once path (default RTN)."""
        for m in model_parts:
            for _, module in tqdm.tqdm(
                list(match_named_modules(m, self.resolved_targets, self.ignore)),
                desc="Calibrating weights",
            ):
                update_weight_zp_scale(module)
        return True

    # ---- template methods for subclasses ------------------------------

    def _process_block(self, blk_idx, blk_name, block, block_inputs, block_mods):
        """Process one decoder block.  Override in subclasses.

        Default (RTN): just ``update_weight_zp_scale`` for each module.

        :param blk_idx: block index
        :param blk_name: block name (e.g. ``layers.0``)
        :param block: the ``nn.Module`` for this decoder block
        :param block_inputs: list of ``(args, kwargs)`` captured microbatches
        :param block_mods: set of target modules inside this block
        """
        for mod in block_mods:
            update_weight_zp_scale(mod)

    def _post_sequential_cleanup(self, model_parts: list[Module]):
        """Called after the sequential loop finishes.  Override for cleanup."""
        pass

    # ---- sequential infrastructure ------------------------------------

    def _build_sequential_blocks(self, model_parts: list[Module]):
        """Discover decoder blocks, build block→modules mapping, register
        capture hook on the first block."""
        for m in model_parts:
            seq_targets = self._infer_sequential_targets(m)
            block_dict = get_layers(seq_targets, m)
            self._sequential_blocks = list(block_dict.items())

            for blk_idx, (blk_name, block) in enumerate(self._sequential_blocks):
                mods = set()
                for _, mod in match_named_modules(block, self.resolved_targets, self.ignore):
                    if getattr_chain(mod, "quantization_scheme.weights", None):
                        mods.add(mod)
                self._block_modules[blk_idx] = mods

            if self._sequential_blocks:
                self.register_hook(
                    self._sequential_blocks[0][1],
                    self._capture_hook,
                    "forward_pre",
                )

        logger.info(
            f"{self.__class__.__name__}: "
            f"{len(self._sequential_blocks)} sequential blocks, "
            f"sequential={self.sequential}"
        )

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
