# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/quantization/quantization/base.py
# licensed under the Apache License 2.0
# modifications:
# - initialize: initialize observers immediately after the model is initialized
# - pre_step: enable observers before forward step
# - post_step: disable observers after forward step, update weights (qdq)
# - finalize: clear observers after all steps are done

import tqdm
import gc
import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules

from modeloptimizer.modifiers import Modifier
from modeloptimizer.modifiers.quantization.mixin import QuantizationMixin
from modeloptimizer.modifiers.quantization.calibration import (
    update_weight_zp_scale,
)
# from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales

__all__ = ["QuantizationModifier"]


class QuantizationModifier(Modifier, QuantizationMixin):

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        """
        Prepare to calibrate activations and weights

        According to the quantization config, a quantization scheme is attached to each
        targeted module. The module's forward call is also overwritten to perform
        quantization to inputs, weights, and outputs.

        Then, according to the module's quantization scheme, observers and calibration
        hooks are added. These hooks are disabled until the modifier starts.
        """
        if not QuantizationMixin.has_config(self):
            raise ValueError("QuantizationModifier requires that quantization fields be specified")
        for m in model_parts:
            QuantizationMixin.initialize_quantization(self, m)

        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        """
        Begin calibrating activations and weights. Calibrate weights only once on start
        """
        for m in model_parts:
            QuantizationMixin.start_calibration(self, m)

        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            QuantizationMixin.end_calibration(self, m)  # keep quantization enabled

            # update weight scales and zero points
            # apply qdq to weights
            named_modules = list(match_named_modules(m, self.resolved_targets, self.ignore))
            for _, module in tqdm.tqdm(named_modules, desc="Calibrating weights"):
                update_weight_zp_scale(module)

        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        for m in model_parts:
            QuantizationMixin.clear_calibration(self, m)

        gc.collect()
        torch.cuda.empty_cache()

        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        return True
