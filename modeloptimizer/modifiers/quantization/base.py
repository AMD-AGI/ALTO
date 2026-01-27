# modified from https://github.com/vllm-project/llm-compressor/blob/f3f14af3ee56e35db7e1faf6da8833f84a570baf/src/llmcompressor/modifiers/quantization/quantization/base.py
# licensed under the Apache License 2.0
# modifications:
# - initialize: initialize observers immediately after the model is initialized
# - pre_step: enable observers before forward step
# - post_step: disable observers after forward step, update weights (qdq)
# - finalize: clear observers after all steps are done

import tqdm
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

    def on_initialize(self, model: Module, **kwargs) -> bool:
        """
        Prepare to calibrate activations and weights

        According to the quantization config, a quantization scheme is attached to each
        targeted module. The module's forward call is also overwritten to perform
        quantization to inputs, weights, and outputs.

        Then, according to the module's quantization scheme, observers and calibration
        hooks are added. These hooks are disabled until the modifier starts.
        """
        if not QuantizationMixin.has_config(self):
            raise ValueError(
                "QuantizationModifier requires that quantization fields be specified"
            )
        QuantizationMixin.initialize_quantization(self, model)

        return True

    def pre_step(self, model: Module, **kwargs):
        """
        Begin calibrating activations and weights. Calibrate weights only once on start
        """
        self.started_ = True
        QuantizationMixin.start_calibration(self, model)

    def post_step(self, model: Module, **kwargs):
        QuantizationMixin.end_calibration(
            self, model)  # keep quantization enabled

        # update weight scales and zero points
        # apply qdq to weights
        named_modules = list(
            match_named_modules(model, self.resolved_targets, self.ignore))
        for _, module in tqdm.tqdm(named_modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

    def on_finalize(self, model: Module, **kwargs) -> bool:
        self.ended_ = True
        QuantizationMixin.clear_calibration(self, model)
