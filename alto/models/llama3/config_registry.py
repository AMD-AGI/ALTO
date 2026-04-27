# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel as llama3_debugmodel_orig,
    llama3_1b as llama3_1b_orig,
    llama3_8b as llama3_8b_orig,
    instella_3b as instella_3b_orig,
)

from alto.components.converter import ModelOptConverter
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader

__all__ = [
    "llama3_debugmodel",
    "llama3_debugmodel_opt",
    "llama3_debugmodel_lpt",
    "llama3_1b",
    "llama3_1b_opt",
    "llama3_1b_lpt",
    "llama3_8b",
    "llama3_8b_opt",
    "llama3_8b_lpt",
    "llama3_1b_gptq",
    "llama3_1b_awq",
    "llama3_8b",
    "llama3_8b_gptq",
    "llama3_8b_rtn",
    "llama3_8b_awq",
    # sparsification
    "llama3_8b_wanda",
    "llama3_8b_sparsegpt",
    "llama3_8b_magnitude",
    "llama3_8b_admm",
    "llama3_8b_alps",
    # structured pruning
    "llama3_8b_wanda_structured",
    "llama3_8b_obs",
    "llama3_8b_admm_structured",
    "llama3_8b_cosine_similarity",
    # other
    "instella_3b",
    "instella_3b_opt",
    "instella_3b_lpt",
]


def llama3_debugmodel() -> Trainer.Config:
    config = llama3_debugmodel_orig()
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 4
    config.training.global_batch_size = 16
    config.training.seq_len = 2048
    config.activation_checkpoint.mode = "none"
    config.debug.seed = 1234
    return config


def llama3_debugmodel_opt() -> Trainer.Config:
    config = llama3_debugmodel()
    config.training.steps = 1
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def llama3_debugmodel_lpt() -> Trainer.Config:
    config = llama3_debugmodel()
    config.training.steps = 10
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe="./alto/models/llama3/configs/lpt_recipe.yaml",)],)
    return config


def llama3_1b() -> Trainer.Config:
    config = llama3_1b_orig()
    config.hf_assets_path = "/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/nlp-pretrained-model/meta-llama/Llama-3.2-1B"
    config.metrics.log_freq = 1
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.global_batch_size = 10
    config.training.seq_len = 8192
    config.dataloader.dataset = "c4_test"
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = "/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/nlp-pretrained-model/meta-llama/Llama-3.2-1B"
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config


def llama3_1b_opt() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def llama3_1b_lpt() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1000
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe="./alto/models/llama3/configs/lpt_recipe.yaml",)],)
    return config


def llama3_8b() -> Trainer.Config:
    config = llama3_8b_orig()
    config.hf_assets_path = "/huggingface/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
    config.metrics.log_freq = 1
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.seq_len = 8192
    config.dataloader.dataset = "c4_test"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 8
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = "/huggingface/hub/models--unsloth--Llama-3.1-8B/snapshots/3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config


def llama3_8b_opt() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def llama3_8b_lpt() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1000
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe="./alto/models/llama3/configs/lpt_recipe.yaml",)],)
    return config

def llama3_1b_gptq() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/gptq_recipe.yaml",),
    ],)
    return config


def llama3_1b_awq() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/awq_recipe.yaml",),
    ],)
    return config


LLAMA3_8B_PATH = "/workspace/Model-Optimizer/models/meta-llama/Llama-3.1-8B"


def llama3_8b() -> Trainer.Config:
    config = llama3_8b_orig()
    config.hf_assets_path = LLAMA3_8B_PATH
    config.metrics.log_freq = 1
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.global_batch_size = 8
    config.training.seq_len = 2048
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = LLAMA3_8B_PATH
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader = HuggingFaceTextDataLoader.Config(dataset="wikitext_test")
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config


def llama3_8b_gptq() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/gptq_recipe.yaml",),
    ],)
    return config


def llama3_8b_rtn() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/rtn_recipe.yaml",),
    ],)
    return config


def llama3_8b_awq() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/awq_recipe.yaml",),
    ],)
    return config


# ======================================================================
# Sparsification configs
# ======================================================================


def llama3_8b_wanda() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/wanda_recipe.yaml",),
    ],)
    return config


def llama3_8b_sparsegpt() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/sparsegpt_recipe.yaml",),
    ],)
    return config


def llama3_8b_magnitude() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/magnitude_recipe.yaml",),
    ],)
    return config


def llama3_8b_admm() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/admm_recipe.yaml",),
    ],)
    return config


def llama3_8b_alps() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/alps_recipe.yaml",),
    ],)
    return config


# ======================================================================
# Structured pruning configs
# ======================================================================


def llama3_8b_wanda_structured() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/wanda_structured_recipe.yaml",),
    ],)
    return config


def llama3_8b_obs() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/obs_recipe.yaml",),
    ],)
    return config


def llama3_8b_admm_structured() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.training.global_batch_size = 128
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/admm_structured_recipe.yaml",),
    ],)
    return config


def llama3_8b_cosine_similarity() -> Trainer.Config:
    config = llama3_8b()
    config.training.steps = 1
    config.optimizer = OptimizersContainer.Config(lr=0.0)
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/cosine_similarity_recipe.yaml",),
    ],)
    return config


# ======================================================================
# Instella configs
# ======================================================================


def instella_3b() -> Trainer.Config:
    config = instella_3b_orig()
    config.hf_assets_path = "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
    config.metrics.log_freq = 1
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.global_batch_size = 10
    config.training.seq_len = 4096
    config.dataloader.dataset = "c4_test"
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config


def instella_3b_opt() -> Trainer.Config:
    config = instella_3b()
    config.training.steps = 1
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def instella_3b_lpt() -> Trainer.Config:
    config = instella_3b()
    config.training.steps = 1000
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe="./alto/models/llama3/configs/lpt_recipe.yaml",)],)
    return config
