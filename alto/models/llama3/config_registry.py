# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.megatron import megatron_dataset_config
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.validator import Validator
from torchtitan.models.common.attention import Attention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.models.llama3 import instella_3b as instella_3b_orig
from torchtitan_recipes.models.llama3 import llama3_1b as llama3_1b_orig
from torchtitan_recipes.models.llama3 import llama3_8b as llama3_8b_orig
from torchtitan_recipes.tests.models.llama3 import (
    llama3_debugmodel as llama3_debugmodel_orig,
)

from alto.components.converter import ModelOptConverter
from alto.models.patcher import ModelPatcher

__all__ = [
    "llama3_debugmodel",
    "llama3_debugmodel_opt",
    "llama3_debugmodel_lpt",
    "llama3_1b",
    "llama3_1b_opt",
    "llama3_1b_lpt",
    "llama3_1b_gptq",
    "llama3_1b_awq",
    "llama3_1b_mx9_wa",
    "llama3_1b_mx6_wa",
    "llama3_8b",
    "llama3_8b_pretrain",
    "llama3_8b_opt",
    "llama3_8b_lpt",
    "llama3_8b_gptq",
    "llama3_8b_rtn",
    "llama3_8b_awq",
    "llama3_8b_wanda",
    "llama3_8b_sparsegpt",
    "llama3_8b_magnitude",
    "llama3_8b_admm",
    "llama3_8b_alps",
    "llama3_8b_wanda_structured",
    "llama3_8b_obs",
    "llama3_8b_admm_structured",
    "llama3_8b_cosine_similarity",
    "instella_3b",
    "instella_3b_opt",
    "instella_3b_lpt",
]

LLAMA3_1B_PATH = (
    "/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/"
    "nlp-pretrained-model/meta-llama/Llama-3.2-1B"
)
LLAMA3_8B_PATH = "/workspace/Model-Optimizer/models/meta-llama/Llama-3.1-8B"
LLAMA3_8B_PRETRAIN_PATH = (
    "/huggingface/hub/models--unsloth--Llama-3.1-8B/snapshots/"
    "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
)
INSTELLA_3B_PATH = (
    "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/"
    "snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
)
C4_TRAIN_IDX = "/workspace/workspace/megatron_dataset/data/c4-train.en_6_text_document.idx"
C4_VAL_IDX = (
    "/workspace/workspace/megatron_dataset/data/"
    "c4-validation-91205-samples.en_text_document.idx"
)
_CONFIGS = "./alto/models/llama3/configs"


def llama3_debugmodel() -> Trainer.Config:
    config = llama3_debugmodel_orig(seq_len=2048, attn_backend="sdpa")
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    return config


def llama3_debugmodel_opt() -> Trainer.Config:
    config = llama3_debugmodel_orig(seq_len=2048, attn_backend="sdpa")
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/recipe.yaml")],
    )
    return config


def llama3_debugmodel_lpt() -> Trainer.Config:
    config = llama3_debugmodel_orig(seq_len=2048, attn_backend="sdpa")
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.profiler.enable_profiling = False
    config.training.steps = 10
    config.training.num_tokens_per_microbatch_per_dp_rank = 4 * 2048
    config.training.num_tokens_per_train_step = 16 * 2048
    config.activation_remat = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/lpt_recipe.yaml")],
    )
    return config


def llama3_1b() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    return config


def llama3_1b_opt() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/recipe.yaml")],
    )
    return config


def llama3_1b_lpt() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1000
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/lpt_recipe.yaml")],
    )
    return config


def llama3_1b_gptq() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/gptq_recipe.yaml")],
    )
    return config


def llama3_1b_awq() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/awq_recipe.yaml")],
    )
    return config


def llama3_1b_mx9_wa() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/mx9_wa_recipe.yaml")],
    )
    return config


def llama3_1b_mx6_wa() -> Trainer.Config:
    config = llama3_1b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_1B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.num_tokens_per_train_step = 10 * 8192
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_1B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/mx6_wa_recipe.yaml")],
    )
    return config


def llama3_8b_pretrain() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PRETRAIN_PATH
    config.dump_folder = "llama3_8b-mi308-pretrain-subset-gbs384-lr1e-4-outputs"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.profiler.enable_profiling = False
    config.training.steps = 5000
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * 8192
    config.training.num_tokens_per_train_step = 384 * 8192
    config.optim.optimizer.optimizers[0].lr = 1e-4
    config.optim.lr_scheduler.min_lr_factor = 0.0
    config.optim.lr_scheduler.warmup_steps = 500
    config.optim.lr_scheduler.decay_ratio = 0.9
    config.optim.lr_scheduler.decay_type = "cosine"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=megatron_dataset_config(C4_TRAIN_IDX)
        ),
    )
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PRETRAIN_PATH,
        initial_load_in_hf=False,
        load_only=True,
    )
    config.validator = None
    config.debug.seed = 1234
    return config


def llama3_8b() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    return config


def llama3_8b_opt() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/recipe.yaml")],
    )
    return config


def llama3_8b_lpt() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=8192)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PRETRAIN_PATH
    config.dump_folder = (
        "llama3_8b-mi308-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-gbs384-lr1e-4-outputs"
    )
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.profiler.enable_profiling = False
    config.training.steps = 5000
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * 8192
    config.training.num_tokens_per_train_step = 384 * 8192
    config.optim.optimizer.optimizers[0].lr = 1e-4
    config.optim.lr_scheduler.min_lr_factor = 0.0
    config.optim.lr_scheduler.warmup_steps = 500
    config.optim.lr_scheduler.decay_ratio = 0.9
    config.optim.lr_scheduler.decay_type = "cosine"
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=megatron_dataset_config(C4_TRAIN_IDX)
        ),
    )
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PRETRAIN_PATH,
        initial_load_in_hf=False,
        load_only=True,
    )
    config.validator = None
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/lpt_recipe.yaml")],
    )
    return config


def llama3_8b_gptq() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/gptq_recipe.yaml")],
    )
    return config


def llama3_8b_rtn() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/rtn_recipe.yaml")],
    )
    return config


def llama3_8b_awq() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/awq_recipe.yaml")],
    )
    return config


def llama3_8b_wanda() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/wanda_recipe.yaml")],
    )
    return config


def llama3_8b_sparsegpt() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/sparsegpt_recipe.yaml")],
    )
    return config


def llama3_8b_magnitude() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(recipe=f"{_CONFIGS}/magnitude_recipe.yaml")
        ],
    )
    return config


def llama3_8b_admm() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/admm_recipe.yaml")],
    )
    return config


def llama3_8b_alps() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/alps_recipe.yaml")],
    )
    return config


def llama3_8b_wanda_structured() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(recipe=f"{_CONFIGS}/wanda_structured_recipe.yaml")
        ],
    )
    return config


def llama3_8b_obs() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/obs_recipe.yaml")],
    )
    return config


def llama3_8b_admm_structured() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 128 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(recipe=f"{_CONFIGS}/admm_structured_recipe.yaml")
        ],
    )
    return config


def llama3_8b_cosine_similarity() -> Trainer.Config:
    config = llama3_8b_orig(seq_len=2048)
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = 8 * 2048
    config.optim.optimizer.optimizers[0].lr = 0.0
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=LLAMA3_8B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(recipe=f"{_CONFIGS}/cosine_similarity_recipe.yaml")
        ],
    )
    return config


def instella_3b() -> Trainer.Config:
    config = instella_3b_orig()
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    seq_len = config.model.max_context_length
    config.hf_assets_path = INSTELLA_3B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 0
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.num_tokens_per_train_step = 10 * seq_len
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=INSTELLA_3B_PATH,
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
    config.debug.seed = 1234
    return config


def instella_3b_opt() -> Trainer.Config:
    config = instella_3b_orig()
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    seq_len = config.model.max_context_length
    config.hf_assets_path = INSTELLA_3B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.num_tokens_per_train_step = 10 * seq_len
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=INSTELLA_3B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/recipe.yaml")],
    )
    return config


def instella_3b_lpt() -> Trainer.Config:
    config = instella_3b_orig()
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    seq_len = config.model.max_context_length
    config.hf_assets_path = INSTELLA_3B_PATH
    config.metrics.log_freq = 1
    config.profiler.enable_profiling = False
    config.training.steps = 1000
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.num_tokens_per_train_step = 10 * seq_len
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
    )
    config.activation_remat = None
    config.checkpointer = CheckpointManager.Config(
        interval=10,
        initial_load_path=INSTELLA_3B_PATH,
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
    config.debug.seed = 1234
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/lpt_recipe.yaml")],
    )
    return config
