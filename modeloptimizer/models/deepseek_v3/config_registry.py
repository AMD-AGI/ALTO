from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel as deepseek_v3_debugmodel_orig,
    deepseek_v3_16b as deepseek_v3_16b_orig,
)

from modeloptimizer.components.converter import ModelOptConverter

__all__ = [
    "deepseek_v3_debugmodel",
    "deepseek_v3_debugmodel_lpt",
    "deepseek_v3_16b",
    "deepseek_v3_16b_lpt",
]


def deepseek_v3_debugmodel() -> Trainer.Config:
    config = deepseek_v3_debugmodel_orig()
    config.profiling.enable_profiling = False
    config.training.steps = 10
    config.training.local_batch_size = 4
    config.training.global_batch_size = 16
    config.training.seq_len = 2048
    config.activation_checkpoint.mode = "none"
    config.debug.seed = 1234
    return config


def deepseek_v3_debugmodel_lpt() -> Trainer.Config:
    config = deepseek_v3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./modeloptimizer/models/deepseek_v3/configs/lpt_recipe.yaml",),
    ],)
    return config


def deepseek_v3_16b() -> Trainer.Config:
    config = deepseek_v3_16b_orig()
    config.hf_assets_path = "/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
    config.dump_folder = "deepseek_v3_16b-outputs"
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.seq_len = 4096
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "c4_test"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = "/huggingface/hub/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.interval = 100
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.activation_checkpoint.mode = "none"
    config.activation_checkpoint.selective_ac_option = "1"
    config.debug.seed = 1234
    return config


def deepseek_v3_16b_lpt() -> Trainer.Config:
    config = deepseek_v3_16b()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./modeloptimizer/models/deepseek/configs/lpt_recipe.yaml",),
    ],)
    return config
