from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.llama3.config_registry import (
    llama3_debugmodel as llama3_debugmodel_orig,
    llama3_1b as llama3_1b_orig,
    instella_3b as instella_3b_orig,
)

from modeloptimizer.components.converter import ModelOptConverter

__all__ = [
    "llama3_debugmodel",
    "llama3_debugmodel_opt",
    "llama3_debugmodel_lpt",
    "llama3_1b",
    "llama3_1b_opt",
    "llama3_1b_lpt",
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
        ModelOptConverter.Config(recipe="./modeloptimizer/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def llama3_debugmodel_lpt() -> Trainer.Config:
    config = llama3_debugmodel()
    config.training.steps = 10
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./modeloptimizer/models/llama3/configs/lpt_recipe.yaml",),
    ],)
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
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = "/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/nlp-pretrained-model/meta-llama/Llama-3.2-1B"
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader = HuggingFaceTextDataLoader.Config(
        dataset="wikitext_test")
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config


def llama3_1b_opt() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./modeloptimizer/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def llama3_1b_lpt() -> Trainer.Config:
    config = llama3_1b()
    config.training.steps = 1000
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./modeloptimizer/models/llama3/configs/lpt_recipe.yaml",),
    ],)
    return config

def instella_3b() -> Trainer.Config:
    config = instella_3b_orig()
    config.hf_assets_path = "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
    config.metrics.log_freq = 1
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.global_batch_size = 10
    config.training.seq_len = 4096
    config.dataloader = HuggingFaceTextDataLoader.Config(dataset="c4_test")
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = True
    config.checkpoint.interval = 10
    config.checkpoint.initial_load_path = "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
    config.checkpoint.initial_load_in_hf = True
    config.validator.enable = True
    config.validator.dataloader = HuggingFaceTextDataLoader.Config(dataset="wikitext_test")
    config.validator.freq = 10
    config.validator.steps = 10
    config.debug.seed = 1234
    return config

def instella_3b_opt() -> Trainer.Config:
    config = instella_3b()
    config.training.steps = 1
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./modeloptimizer/models/llama3/configs/recipe.yaml",),
    ],)
    return config


def instella_3b_lpt() -> Trainer.Config:
    config = instella_3b()
    config.training.steps = 1000
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./modeloptimizer/models/llama3/configs/lpt_recipe.yaml",),
    ],)
    return config
