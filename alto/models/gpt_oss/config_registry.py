# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""GPT-OSS configs.

The debug config comes from the upstream recipe. The 20B configs keep ALTO's
Hugging Face path, batch, learning-rate schedule, and Megatron data settings.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.megatron import megatron_dataset_config
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.gpt_oss import build_model_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.tests.models.gpt_oss import gpt_oss_debugmodel as _gpt_oss_debugmodel

from alto.components.converter import ModelOptConverter

__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_lpt",
    "gpt_oss_20b",
    "gpt_oss_20b_pretrain",
    "gpt_oss_20b_lpt",
]

_LPT_RECIPE = "./alto/models/gpt_oss/configs/lpt_recipe.yaml"
GPT_OSS_20B_PATH = (
    "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee/"
)
C4_TRAIN_IDX = "/workspace/workspace/megatron_dataset/data/c4-train.en_6_text_document.idx"
C4_VAL_IDX = (
    "/workspace/workspace/megatron_dataset/data/"
    "c4-validation-91205-samples.en_text_document.idx"
)


def _source(name: str):
    if name.endswith(".idx"):
        return megatron_dataset_config(name)
    return DATASETS[name]


def _dataloader(name: str, *, micro_tokens: int, shuffle: bool, bound_documents: bool):
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=_source(name)),
        shuffle=shuffle,
        max_num_documents=(
            c4_document_capacity(micro_tokens) if bound_documents else None
        ),
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


def _set_dataset(config: Trainer.Config, name: str, *, bound_documents: bool) -> None:
    config.dataloader = _dataloader(
        name,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
        shuffle=True,
        bound_documents=bound_documents,
    )


def _validator(name: str, *, freq: int, steps: int, micro_tokens: int) -> Validator.Config:
    return Validator.Config(
        freq=freq,
        steps=steps,
        dataloader=_dataloader(
            name,
            micro_tokens=micro_tokens,
            shuffle=False,
            bound_documents=True,
        ),
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


def _gpt_oss_20b(seq_len: int, parallelism: ParallelismConfig) -> Trainer.Config:
    config = _gpt_oss_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config.model = build_model_config("20b", seq_len=seq_len, parallelism=parallelism)
    config.loss.loss_fn.global_vocab_size = decoder_vocab_size(config.model)
    config.training.max_context_length = config.model.max_context_length
    return config


def gpt_oss_debugmodel() -> Trainer.Config:
    config = _gpt_oss_debugmodel(
        seq_len=2048,
        parallelism=ParallelismConfig(expert_parallel_degree=1),
    )
    return _prepare(config, steps=10, local_batch=4, global_batch=16, log_freq=None)


def gpt_oss_debugmodel_lpt() -> Trainer.Config:
    return _attach(gpt_oss_debugmodel())


def gpt_oss_20b() -> Trainer.Config:
    config = _prepare(
        _gpt_oss_20b(
            8192,
            ParallelismConfig(expert_parallel_degree=1, tensor_parallel_degree=1),
        ),
        steps=0,
        local_batch=1,
    )
    config.hf_assets_path = GPT_OSS_20B_PATH
    config.dump_folder = "gpt_oss_20b-outputs"
    config.metrics.enable_tensorboard = True
    _set_dataset(config, "c4_test", bound_documents=True)
    config.checkpointer = CheckpointManager.Config(
        interval=100,
        initial_load_path=GPT_OSS_20B_PATH,
        initial_load_in_hf=True,
        initial_load_in_hf_quantized=True,
    )
    config.validator = _validator(
        "wikitext_test",
        freq=10,
        steps=10,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
    )
    return config


def gpt_oss_20b_pretrain() -> Trainer.Config:
    config = _prepare(
        _gpt_oss_20b(
            8192,
            ParallelismConfig(expert_parallel_degree=8, tensor_parallel_degree=1),
        ),
        steps=1_200_000,
        local_batch=1,
        global_batch=16,
    )
    config.hf_assets_path = GPT_OSS_20B_PATH
    config.dump_folder = "gpt_oss_20b-pretrain-subset-lr4e-4-outputs"
    config.training.disable_cuda_graphs = True
    optimizer = config.optim.optimizer.optimizers[0]
    optimizer.lr = 4e-4
    optimizer.betas = (0.9, 0.95)
    optimizer.eps = 1e-5
    config.optim.lr_scheduler.min_lr_factor = 0.1
    config.optim.lr_scheduler.warmup_steps = 128
    config.optim.lr_scheduler.decay_ratio = 1 - 128 / config.training.steps
    config.optim.lr_scheduler.decay_type = "cosine"
    config.metrics.enable_tensorboard = True
    config.metrics.enable_wandb = True
    _set_dataset(config, C4_TRAIN_IDX, bound_documents=True)
    config.checkpointer = CheckpointManager.Config(
        interval=1000,
        keep_latest_k=2,
        load_only=True,
    )
    config.validator = _validator(
        C4_VAL_IDX,
        freq=768,
        steps=64,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
    )
    return config


def gpt_oss_20b_lpt() -> Trainer.Config:
    config = gpt_oss_20b_pretrain()
    config.dump_folder = (
        "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-uos-lr4e-4-bitrefactor-outputs"
    )
    return _attach(config)
