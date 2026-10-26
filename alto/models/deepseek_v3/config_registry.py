# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""DeepSeek-V3 configs.

Upstream recipes supply the model skeleton. These functions restore ALTO's
sequence length, batch size, Hugging Face checkpoint, and validator settings.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.trainer import Trainer
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_16b as _deepseek_v3_16b,
)
from torchtitan_recipes.tests.models.deepseek_v3 import (
    deepseek_v3_debugmodel as _deepseek_v3_debugmodel,
)

from alto.components.converter import ModelOptConverter

__all__ = [
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_lpt",
    "deepseek_v3_16b",
    "deepseek_v3_16b_lpt",
]

_LPT_RECIPE = "./alto/models/deepseek_v3/configs/lpt_recipe.yaml"
DEEPSEEK_V3_16B_PATH = (
    "/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/"
    "521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
)


def _dataloader(name: str, *, micro_tokens: int, shuffle: bool):
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS[name]),
        shuffle=shuffle,
        max_num_documents=c4_document_capacity(micro_tokens),
    )


def _set_local_batch(config: Trainer.Config, local_batch: int) -> None:
    seq_len = config.training.max_context_length
    micro_tokens = local_batch * seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = micro_tokens
    if getattr(config.dataloader, "max_num_documents", None) is not None:
        config.dataloader.max_num_documents = c4_document_capacity(micro_tokens)


def _set_global_batch(config: Trainer.Config, global_batch: int) -> None:
    config.training.num_tokens_per_train_step = (
        global_batch * config.training.max_context_length
    )


def _prepare(
    config: Trainer.Config,
    *,
    steps: int,
    local_batch: int,
    global_batch: int | None = None,
    log_freq: int | None = 1,
) -> Trainer.Config:
    config.profiler.enable_profiling = False
    config.debug.seed = 1234
    config.activation_remat = None
    config.training.steps = steps
    if log_freq is not None:
        config.metrics.log_freq = log_freq
    _set_local_batch(config, local_batch)
    if global_batch is not None:
        _set_global_batch(config, global_batch)
    return config


def _attach(config: Trainer.Config) -> Trainer.Config:
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=_LPT_RECIPE)]
    )
    config.training.disable_cuda_graphs = True
    return config


def deepseek_v3_debugmodel() -> Trainer.Config:
    config = _deepseek_v3_debugmodel(seq_len=2048, parallelism=ParallelismConfig())
    return _prepare(config, steps=10, local_batch=4, global_batch=16, log_freq=None)


def deepseek_v3_debugmodel_lpt() -> Trainer.Config:
    return _attach(deepseek_v3_debugmodel())


def deepseek_v3_16b() -> Trainer.Config:
    config = _prepare(
        _deepseek_v3_16b(
            seq_len=4096,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=1,
                expert_parallel_degree=1,
            ),
        ),
        steps=0,
        local_batch=1,
    )
    config.hf_assets_path = DEEPSEEK_V3_16B_PATH
    config.dump_folder = "deepseek_v3_16b-outputs"
    config.metrics.enable_tensorboard = True
    micro_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    config.dataloader = _dataloader("c4_test", micro_tokens=micro_tokens, shuffle=True)
    config.checkpointer = CheckpointManager.Config(
        interval=100,
        initial_load_path=DEEPSEEK_V3_16B_PATH,
        initial_load_in_hf=True,
    )
    config.validator = Validator.Config(
        freq=10,
        steps=10,
        dataloader=_dataloader("wikitext_test", micro_tokens=micro_tokens, shuffle=False),
    )
    return config


def deepseek_v3_16b_lpt() -> Trainer.Config:
    return _attach(deepseek_v3_16b())
