# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.gpt_oss.config_registry import (
    gpt_oss_debugmodel as gpt_oss_debugmodel_orig,
    gpt_oss_20b as gpt_oss_20b_orig,
)

from alto.components.converter import ModelOptConverter
from alto.components.m_adam import MAdamOptimizersContainer

__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_lpt",
    "gpt_oss_20b",
    "gpt_oss_20b_pretrain",
    "gpt_oss_20b_lpt",
    "gpt_oss_20b_madam",
]


def gpt_oss_debugmodel() -> Trainer.Config:
    config = gpt_oss_debugmodel_orig()
    config.profiling.enable_profiling = False
    config.training.steps = 10
    config.training.local_batch_size = 4
    config.training.global_batch_size = 16
    config.training.seq_len = 2048
    config.activation_checkpoint.mode = "none"
    config.debug.seed = 1234
    return config


def gpt_oss_debugmodel_lpt() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config


def gpt_oss_20b() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.dump_folder = "gpt_oss_20b-outputs"
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.seq_len = 8192
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "c4_test"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_in_hf_quantized = True
    config.checkpoint.interval = 100
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.activation_checkpoint.mode = "none"
    config.activation_checkpoint.selective_ac_option = "1"
    config.debug.seed = 1234
    return config


def gpt_oss_20b_pretrain() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.dump_folder = "gpt_oss_20b-pretrain-subset-lr4e-4-outputs"
    config.profiling.enable_profiling = False
    config.training.steps = 1200000
    config.training.local_batch_size = 1
    config.training.global_batch_size = 16
    config.training.seq_len = 8192
    config.optimizer.lr = 4e-4
    config.optimizer.weight_decay = 0.1
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-5
    config.lr_scheduler.min_lr_factor = 0.1
    config.lr_scheduler.warmup_steps = 128
    config.lr_scheduler.decay_ratio = 1 - 128 / config.training.steps
    config.lr_scheduler.decay_type = "cosine"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "megatron"
    config.dataloader.dataset_path = "/workspace/workspace/megatron_dataset/data/c4-train.en_6_text_document.idx"
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 4
    config.checkpoint.enable = True
    config.checkpoint.interval = 1000
    config.checkpoint.keep_latest_k = 2
    config.validator.enable = True
    config.validator.dataloader.dataset = "megatron"
    config.validator.dataloader.dataset_path = "/workspace/workspace/megatron_dataset/data/c4-validation-91205-samples.en_text_document.idx"
    config.validator.freq = 768
    config.validator.steps = 64
    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "1"
    config.debug.seed = 1234
    return config

def gpt_oss_20b_pretrain_c4_megatron() -> Trainer.Config:
    """gpt_oss_20b_pretrain using HuggingFace C4 dataset (bf16 baseline)."""
    config = gpt_oss_20b_pretrain()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-bf16-c4-outputs"
    config.dataloader.dataset_path = "/data/c4-train.en_6_text_document.idx"
    config.validator.dataloader.dataset_path = "/data/c4-validation-91205-samples.en_text_document.idx"
    config.checkpoint.initial_load_path = None      # fresh run: do NOT load any checkpoint
    config.checkpoint.initial_load_in_hf = False
    config.checkpoint.initial_load_in_hf_quantized = False
    config.checkpoint.last_save_model_only = False   # save full ckpt at final step (model+optim+dataloader) so training can resume
    return config

def gpt_oss_20b_lpt() -> Trainer.Config:
    config = gpt_oss_20b_pretrain_c4_megatron()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config


def gpt_oss_20b_madam() -> Trainer.Config:
    """gpt_oss_20b pretrain (plain BF16, no quantization) with the custom
    ``m_adam`` optimizer.

    ``lr`` maps to m_adam's additive (AdamW) branch ``lr_m`` and is driven by
    the usual LR scheduler; ``lr_e`` is the multiplicative (exponent) branch
    learning rate (tied to ``lr_m`` here via ``tie_e_to_m``).
    """
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-madam-outputs"
    config.optimizer = MAdamOptimizersContainer.Config(
        lr=1e-4,
        lr_e=1e-4,
        tie_e_to_m=True,
        beta1=0.9,
        beta2=0.95,
        eps=1e-5,
        weight_decay_m=0.1,
    )
    return config
