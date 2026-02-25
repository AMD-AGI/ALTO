from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.llama3.config_registry import llama3_debugmodel as llama3_debugmodel_orig

from modeloptimizer.components.converter import ModelOptConverter

__all__ = ["llama3_debugmodel", "llama3_debugmodel_opt"]


def llama3_debugmodel() -> Trainer.Config:
    config = llama3_debugmodel_orig()
    config.activation_checkpoint.mode = "none"
    return config


def llama3_debugmodel_opt() -> Trainer.Config:
    config = llama3_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./modeloptimizer/models/llama3/configs/recipe.yaml",),
    ],)
    return config
