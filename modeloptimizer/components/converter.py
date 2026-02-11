# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.components.quantization import (
    QuantizationConverter,)
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

from modeloptimizer.config import Recipe


class ModelOptConverter(QuantizationConverter):

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)
        self.recipe = Recipe.create_instance(job_config.modeloptimizer.recipe)

    def convert(self, model: nn.Module):
        from torchtitan.components.checkpoint import CheckpointManager
        CheckpointManager.allow_partial_load = True

    def pre_step(self, model_parts: list[nn.Module]):
        for modifier in self.recipe.modifiers:
            modifier.pre_step(model_parts)

    def post_optimizer_hook(self, model_parts: list[nn.Module], **kwargs):
        for modifier in self.recipe.modifiers:
            modifier.post_step(model_parts, **kwargs)

    def post_initialization(self, model_parts: list[nn.Module]):
        for modifier in self.recipe.modifiers:
            modifier.initialize(model_parts)

    def finalize(self, model_parts: list[nn.Module]):
        for modifier in self.recipe.modifiers:
            modifier.finalize(model_parts)


register_model_converter(ModelOptConverter, "modeloptimizer")
