# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING
from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.components.model_converter import ModelConverter
from torchtitan.config import Configurable

from torchtitan.distributed import ParallelismContext

from alto.config import Recipe

if TYPE_CHECKING:
    from torchtitan.models.base import BaseModel


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
        parallelism_context: ParallelismContext | None,
        model_compile_enabled: bool,
    ):
        # if parallelism_context is not None:
        #     assert (not parallelism_context.dp_enabled), "Model optimizer does not yet support data parallelism"
        #     assert (not parallelism_context.tp_enabled), "Model optimizer does not yet support tensor parallelism"
        #     assert (not parallelism_context.cp_enabled), "Model optimizer does not yet support context parallelism"

        self.recipe = Recipe.create_instance(config.recipe)

    def convert(self, model: nn.Module):
        from torchtitan.components.checkpointer import CheckpointManager
        CheckpointManager.allow_partial_load = True

        for modifier in self.recipe.modifiers:
            modifier.convert(model)

    def convert_config(self, model_config: "BaseModel.Config"):
        from torchtitan.models.common.linear import Linear

        # Modifiers replace or wrap weights (fake quantization, tensor subclasses,
        # pruning masks), which the in-place WGRAD autograd function cannot handle.
        for _fqn, config, _parent, _attr in model_config.traverse(Linear.Config):
            config.inplace_wgrad_accum = False

        for modifier in self.recipe.modifiers:
            modifier.convert_config(model_config)

    def pre_step(self, model_parts: list[nn.Module], **kwargs):
        for modifier in self.recipe.modifiers:
            modifier.pre_step(model_parts, **kwargs)

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
