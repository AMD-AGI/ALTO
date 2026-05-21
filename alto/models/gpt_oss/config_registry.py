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

GPT_OSS_20B_HF_PATH = (
    "/wekafs/hanwang2/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
GPT_OSS_C4_TRAIN_IDX = (
    "/wekafs/hanwang2/workspace/megatron_dataset/data/"
    "c4-train.en_6_text_document.idx"
)
GPT_OSS_C4_VALID_IDX = (
    "/wekafs/hanwang2/workspace/megatron_dataset/data/c4-validation-91205-samples.en_text_document.idx"
)
GPT_OSS_RUN_ROOT = "/wekafs/hanwang2/workspace/alto_runs"

__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_lpt",
    "gpt_oss_debugmodel_nvfp4_lpt",
    "gpt_oss_20b",
    "gpt_oss_20b_pretrain",
    "gpt_oss_20b_lpt",
    "gpt_oss_20b_nvfp4_lpt",
    "gpt_oss_20b_nvfp4_r8_lpt",
    "gpt_oss_20b_nvfp4_r8_nosr_lpt",
    "gpt_oss_20b_nvfp4_dense_only_lpt",
    "gpt_oss_20b_nvfp4_grouped_only_lpt",
    "gpt_oss_20b_nvfp4_had_lpt",
    "gpt_oss_20b_nvfp4_pts_lpt",
    "gpt_oss_20b_nvfp4_smoke",
    "gpt_oss_20b_nvfp4_iso_h0_sr_none_lpt",
    "gpt_oss_20b_nvfp4_iso_h0_sr_dgrad_lpt",
    "gpt_oss_20b_nvfp4_iso_h0_sr_wgrad_lpt",
    "gpt_oss_20b_nvfp4_iso_h0_sr_all_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_none_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_dgrad_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_wgrad_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_all_lpt",
    "gpt_oss_20b_nvfp4_iso_h0_sr_all_pts_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_dgrad_pts_lpt",
    "gpt_oss_20b_nvfp4_iso_h1_sr_wgrad_pts_lpt",
    "gpt_oss_20b_nvfp4_grouped_early_lpt",
    "gpt_oss_20b_nvfp4_grouped_middle_lpt",
    "gpt_oss_20b_nvfp4_grouped_late_lpt",
    "gpt_oss_20b_nvfp4_grouped_layer0_lpt",
    "gpt_oss_20b_mxfp4_grouped_early_lpt",
    "gpt_oss_20b_mxfp4_grouped_middle_lpt",
    "gpt_oss_20b_mxfp4_grouped_late_lpt",
    "gpt_oss_20b_mxfp4_grouped_layer0_lpt",
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


def gpt_oss_debugmodel_nvfp4_lpt() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/lpt_nvfp4_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_20b() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = GPT_OSS_20B_HF_PATH
    config.dump_folder = f"{GPT_OSS_RUN_ROOT}/gpt_oss_20b-outputs"
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
    config.checkpoint.initial_load_path = GPT_OSS_20B_HF_PATH
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
    config.hf_assets_path = GPT_OSS_20B_HF_PATH
    config.dump_folder = (
        f"{GPT_OSS_RUN_ROOT}/gpt_oss_20b-pretrain-subset-lr4e-4-outputs"
    )
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
    config.dataloader.dataset_path = GPT_OSS_C4_TRAIN_IDX
    config.parallelism.expert_parallel_degree = 4
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 4
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = GPT_OSS_20B_HF_PATH
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_in_hf_quantized = True
    config.checkpoint.interval = 1000
    config.checkpoint.keep_latest_k = 2
    config.validator.enable = True
    config.validator.dataloader.dataset = "megatron"
    config.validator.dataloader.dataset_path = GPT_OSS_C4_VALID_IDX
    config.validator.freq = 768
    config.validator.steps = 64
    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "1"
    config.debug.seed = 1234
    return config


def gpt_oss_20b_lpt() -> Trainer.Config:
    config = gpt_oss_20b_pretrain()
    config.dump_folder = (
        f"{GPT_OSS_RUN_ROOT}/"
        "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-outputs"
    )
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config


def _gpt_oss_20b_nvfp4_lpt_config(recipe: str, dump_suffix: str) -> Trainer.Config:
    config = gpt_oss_20b_pretrain()
    config.dump_folder = f"{GPT_OSS_RUN_ROOT}/{dump_suffix}"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe=recipe,
        ),
    ],)
    return config


def gpt_oss_20b_nvfp4_r8_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/lpt_nvfp4_20b_r8_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-r8-w2d-tail3-sr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_r8_nosr_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_r8_nosr_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-r8-w2d-tail3-nosr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_dense_only_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_dense_only_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-dense-only-w2d-tail3-sr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_grouped_only_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_grouped_only_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-grouped-only-w2d-tail3-sr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_had_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_had_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-had-w2d-tail3-sr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_pts_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_pts_recipe.yaml",
        "gpt_oss_20b-pretrain-subset-nvfp4-pts-w2d-tail3-sr-lr4e-4-outputs",
    )


def gpt_oss_20b_nvfp4_lpt() -> Trainer.Config:
    return gpt_oss_20b_nvfp4_r8_lpt()


def gpt_oss_20b_nvfp4_smoke() -> Trainer.Config:
    config = gpt_oss_20b_nvfp4_r8_lpt()
    config.dump_folder = f"{GPT_OSS_RUN_ROOT}/gpt_oss_20b-nvfp4-r8-smoke"
    config.training.steps = 1
    config.metrics.log_freq = 1
    config.checkpoint.enable = False
    config.validator.enable = False
    return config


def _gpt_oss_20b_nvfp4_iso_lpt_config(name: str) -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        f"./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_iso_{name}_recipe.yaml",
        f"gpt_oss_20b-nvfp4-iso-{name}-10step",
    )


def gpt_oss_20b_nvfp4_iso_h0_sr_none_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h0_sr_none")


def gpt_oss_20b_nvfp4_iso_h0_sr_dgrad_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h0_sr_dgrad")


def gpt_oss_20b_nvfp4_iso_h0_sr_wgrad_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h0_sr_wgrad")


def gpt_oss_20b_nvfp4_iso_h0_sr_all_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h0_sr_all")


def gpt_oss_20b_nvfp4_iso_h1_sr_none_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_none")


def gpt_oss_20b_nvfp4_iso_h1_sr_dgrad_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_dgrad")


def gpt_oss_20b_nvfp4_iso_h1_sr_wgrad_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_wgrad")


def gpt_oss_20b_nvfp4_iso_h1_sr_all_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_all")


def gpt_oss_20b_nvfp4_iso_h0_sr_all_pts_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h0_sr_all_pts")


def gpt_oss_20b_nvfp4_iso_h1_sr_dgrad_pts_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_dgrad_pts")


def gpt_oss_20b_nvfp4_iso_h1_sr_wgrad_pts_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_iso_lpt_config("h1_sr_wgrad_pts")


def gpt_oss_20b_nvfp4_grouped_early_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_grouped_early_recipe.yaml",
        "gpt_oss_20b-nvfp4-grouped-early",
    )


def gpt_oss_20b_nvfp4_grouped_middle_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_grouped_middle_recipe.yaml",
        "gpt_oss_20b-nvfp4-grouped-middle",
    )


def gpt_oss_20b_nvfp4_grouped_late_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_grouped_late_recipe.yaml",
        "gpt_oss_20b-nvfp4-grouped-late",
    )


def gpt_oss_20b_nvfp4_grouped_layer0_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_grouped_layer0_recipe.yaml",
        "gpt_oss_20b-nvfp4-grouped-layer0",
    )


def gpt_oss_20b_mxfp4_grouped_early_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_mxfp4_20b_grouped_early_recipe.yaml",
        "gpt_oss_20b-mxfp4-grouped-early",
    )


def gpt_oss_20b_mxfp4_grouped_middle_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_mxfp4_20b_grouped_middle_recipe.yaml",
        "gpt_oss_20b-mxfp4-grouped-middle",
    )


def gpt_oss_20b_mxfp4_grouped_late_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_mxfp4_20b_grouped_late_recipe.yaml",
        "gpt_oss_20b-mxfp4-grouped-late",
    )


def gpt_oss_20b_mxfp4_grouped_layer0_lpt() -> Trainer.Config:
    return _gpt_oss_20b_nvfp4_lpt_config(
        "./alto/models/gpt_oss/configs/experimental/lpt_mxfp4_20b_grouped_layer0_recipe.yaml",
        "gpt_oss_20b-mxfp4-grouped-layer0",
    )
