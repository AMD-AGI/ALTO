# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_16b as deepseek_v3_16b_orig,
)
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel as deepseek_v3_debugmodel_orig,
)

from alto.components.converter import ModelOptConverter

__all__ = [
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_lpt",
    "deepseek_v3_16b",
    "deepseek_v3_16b_lpt",
]

DEEPSEEK_V3_16B_PATH = (
    "/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/"
    "521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
)


def deepseek_v3_debugmodel() -> Trainer.Config:
    config = deepseek_v3_debugmodel_orig(seq_len=2048)
    config.profiler.enable_profiling = False
    config.training.steps = 10
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    return config


def deepseek_v3_debugmodel_lpt() -> Trainer.Config:
    config = deepseek_v3_debugmodel_orig(seq_len=2048)
    config.profiler.enable_profiling = False
    config.training.steps = 10
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(
                recipe="./alto/models/deepseek_v3/configs/lpt_recipe.yaml",
            )
        ],
    )
    return config


def deepseek_v3_16b() -> Trainer.Config:
    config = deepseek_v3_16b_orig(
        seq_len=4096,
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
    )
    config.hf_assets_path = DEEPSEEK_V3_16B_PATH
    config.dump_folder = "deepseek_v3_16b-outputs"
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.disable_cuda_graphs = True
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.checkpointer = CheckpointManager.Config(
        interval=100,
        initial_load_path=DEEPSEEK_V3_16B_PATH,
        initial_load_in_hf=True,
    )
    config.validator = Validator.Config(
        freq=10,
        steps=10,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["wikitext_test"]),
            shuffle=False,
        ),
    )
    config.activation_remat = None
    config.debug.seed = 1234
    return config


def deepseek_v3_16b_lpt() -> Trainer.Config:
    config = deepseek_v3_16b_orig(
        seq_len=4096,
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            tensor_parallel_degree=1,
        ),
    )
    config.hf_assets_path = DEEPSEEK_V3_16B_PATH
    config.dump_folder = "deepseek_v3_16b-outputs"
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.disable_cuda_graphs = True
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.checkpointer = CheckpointManager.Config(
        interval=100,
        initial_load_path=DEEPSEEK_V3_16B_PATH,
        initial_load_in_hf=True,
    )
    config.validator = Validator.Config(
        freq=10,
        steps=10,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["wikitext_test"]),
            shuffle=False,
        ),
    )
    config.activation_remat = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(
                recipe="./alto/models/deepseek_v3/configs/lpt_recipe.yaml",
            )
        ],
    )
    return config
