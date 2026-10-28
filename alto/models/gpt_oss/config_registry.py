# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.megatron import megatron_dataset_config
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.validator import Validator
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.gpt_oss import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.gpt_oss import (
    gpt_oss_debugmodel as gpt_oss_debugmodel_orig,
)

from alto.components.converter import ModelOptConverter

__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_lpt",
    "gpt_oss_20b",
    "gpt_oss_20b_pretrain",
    "gpt_oss_20b_lpt",
]

GPT_OSS_20B_PATH = (
    "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee/"
)
C4_TRAIN_IDX = "/workspace/workspace/megatron_dataset/data/c4-train.en_6_text_document.idx"
C4_VAL_IDX = (
    "/workspace/workspace/megatron_dataset/data/"
    "c4-validation-91205-samples.en_text_document.idx"
)


def gpt_oss_debugmodel() -> Trainer.Config:
    config = gpt_oss_debugmodel_orig(seq_len=2048)
    config.profiler.enable_profiling = False
    config.training.steps = 10
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    return config


def gpt_oss_debugmodel_lpt() -> Trainer.Config:
    config = gpt_oss_debugmodel_orig(seq_len=2048)
    config.profiler.enable_profiling = False
    config.training.steps = 10
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(
                recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",
            )
        ],
    )
    return config


def gpt_oss_20b() -> Trainer.Config:
    seq_len = 8192
    parallelism = ParallelismConfig(
        expert_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    model = build_model_config("20b", seq_len=seq_len, parallelism=parallelism)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        hf_assets_path=GPT_OSS_20B_PATH,
        dump_folder="gpt_oss_20b-outputs",
        profiler=Profiler.Config(enable_profiling=False),
        metrics=MetricsProcessor.Config(log_freq=1, enable_tensorboard=True),
        model=model,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2000,
                decay_ratio=0.8,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            # Varlen attention has no C4 document cap in this eval config.
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            max_context_length=model.max_context_length,
            steps=0,
        ),
        parallelism=parallelism,
        checkpointer=CheckpointManager.Config(
            interval=100,
            initial_load_path=GPT_OSS_20B_PATH,
            initial_load_in_hf=True,
            initial_load_in_hf_quantized=True,
        ),
        activation_remat=None,
        debug=DebugConfig(seed=1234),
        validator=Validator.Config(
            freq=10,
            steps=10,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["wikitext_test"]),
                shuffle=False,
            ),
        ),
    )


def gpt_oss_20b_pretrain() -> Trainer.Config:
    seq_len = 8192
    steps = 1_200_000
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
        tensor_parallel_degree=1,
    )
    model = build_model_config("20b", seq_len=seq_len, parallelism=parallelism)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        hf_assets_path=GPT_OSS_20B_PATH,
        dump_folder="gpt_oss_20b-pretrain-subset-lr4e-4-outputs",
        profiler=Profiler.Config(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=True,
            enable_wandb=True,
        ),
        model=model,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=megatron_dataset_config(C4_TRAIN_IDX)
            ),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=4e-4,
                        betas=(0.9, 0.95),
                        eps=1e-5,
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=128,
                decay_ratio=1 - 128 / steps,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            num_tokens_per_train_step=16 * seq_len,
            max_context_length=model.max_context_length,
            steps=steps,
        ),
        parallelism=parallelism,
        checkpointer=CheckpointManager.Config(
            interval=1000,
            keep_latest_k=2,
            load_only=True,
            initial_load_model_only=False,
        ),
        activation_remat=None,
        debug=DebugConfig(seed=1234),
        validator=Validator.Config(
            freq=768,
            steps=64,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=megatron_dataset_config(C4_VAL_IDX)
                ),
                shuffle=False,
            ),
        ),
    )


def gpt_oss_20b_lpt() -> Trainer.Config:
    seq_len = 8192
    steps = 1_200_000
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
        tensor_parallel_degree=1,
    )
    model = build_model_config("20b", seq_len=seq_len, parallelism=parallelism)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        hf_assets_path=GPT_OSS_20B_PATH,
        dump_folder=(
            "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-uos-"
            "lr4e-4-bitrefactor-outputs"
        ),
        profiler=Profiler.Config(enable_profiling=False),
        metrics=MetricsProcessor.Config(
            log_freq=1,
            enable_tensorboard=True,
            enable_wandb=True,
        ),
        model=model,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=megatron_dataset_config(C4_TRAIN_IDX)
            ),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[
                    AdamW.Config(
                        pattern=r".*",
                        lr=4e-4,
                        betas=(0.9, 0.95),
                        eps=1e-5,
                    )
                ]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=128,
                decay_ratio=1 - 128 / steps,
                decay_type="cosine",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            num_tokens_per_train_step=16 * seq_len,
            max_context_length=model.max_context_length,
            steps=steps,
        ),
        parallelism=parallelism,
        checkpointer=CheckpointManager.Config(
            interval=1000,
            keep_latest_k=2,
            load_only=True,
            initial_load_model_only=False,
        ),
        activation_remat=None,
        debug=DebugConfig(seed=1234),
        validator=Validator.Config(
            freq=768,
            steps=64,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=megatron_dataset_config(C4_VAL_IDX)
                ),
                shuffle=False,
            ),
        ),
        model_converters=ModelConvertersContainer.Config(
            converters=[
                ModelOptConverter.Config(
                    recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",
                )
            ],
        ),
    )
