# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.protocols.model_converter import ModelConverter
from torchtitan.config import Configurable

from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from alto.config import Recipe


class ModelOptConverter(ModelConverter, Configurable):

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        recipe: str = ""
        """
        Path to the model optimizer recipe file.
        """

    def __init__(
        self,
        config: Config,
        *,
        parallel_dims: ParallelDims,
        model_compile_enabled: bool,
    ):
        # if parallel_dims is not None:
        #     assert (not parallel_dims.dp_enabled), "Model optimizer does not yet support data parallelism"
        #     assert (not parallel_dims.tp_enabled), "Model optimizer does not yet support tensor parallelism"
        #     assert (not parallel_dims.cp_enabled), "Model optimizer does not yet support context parallelism"

        self.recipe = Recipe.create_instance(config.recipe)

    def convert(self, model: nn.Module):
        from torchtitan.components.checkpoint import CheckpointManager
        CheckpointManager.allow_partial_load = True

        for modifier in self.recipe.modifiers:
            modifier.convert(model)

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

    @property
    def requires_replay_buffer(self) -> bool:
        return any(modifier.requires_replay_buffer for modifier in self.recipe.modifiers)

    @property
    def requires_training_mode(self) -> bool:
        return any(modifier.requires_training_mode for modifier in self.recipe.modifiers)
