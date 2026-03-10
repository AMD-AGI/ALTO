from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.gpt_oss.config_registry import (
    gpt_oss_debugmodel as gpt_oss_debugmodel_orig,
    gpt_oss_20b as gpt_oss_20b_orig,
)


__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_20b",
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


def gpt_oss_20b() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = "/group/ossmodelzoo/hanwang2/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.dump_folder = "gpt_oss_20b-outputs"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "c4_test"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 8
    config.checkpoint.enable = False
    config.checkpoint.interval = 100
    config.validator.enable = False
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.activation_checkpoint.mode = "selective"
    config.activation_checkpoint.selective_ac_option = "1"
    config.debug.seed = 1234
    return config
