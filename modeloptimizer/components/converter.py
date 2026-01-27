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
        self.recipe = Recipe.create_instance(job_config.model_optimizer.recipe)

    def convert(self, model: nn.Module):
        # TODO untie word embeddings
        pass

    def pre_step(self, model: nn.Module | list[nn.Module]):
        for modifier in self.recipe.modifiers:
            if not isinstance(model, list):
                model = [model]
            for m in model:
                modifier.pre_step(m)

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        for modifier in self.recipe.modifiers:
            if not isinstance(model, list):
                model = [model]
            for m in model:
                modifier.post_step(m)

    def post_initialization(self, model: nn.Module):
        for modifier in self.recipe.modifiers:
            modifier.initialize(model)

    def finalize(self, model: nn.Module | list[nn.Module]):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for modifier in self.recipe.modifiers:
                modifier.finalize(m)


register_model_converter(ModelOptConverter, "modeloptimizer")
