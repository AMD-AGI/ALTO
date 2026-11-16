# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.components.validator import Validator
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.experiments.transformers_modeling_backend import build_model_config
from torchtitan.experiments.transformers_modeling_backend.tokenizer import HFBackendTokenizer
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.observability.profiler import Profiler
from torchtitan.trainer import Trainer

from alto.components.converter import ModelOptConverter

__all__ = [
    "hf_backend_mx9_wa",
    "hf_backend_mx6_wa",
    "hf_backend_bf16",
]

MX6_RECIPE = "./alto/models/transformers_modeling_backend/configs/mx6_wa_recipe.yaml"
MX9_RECIPE = "./alto/models/transformers_modeling_backend/configs/mx9_wa_recipe.yaml"


def hf_backend(
    seq_len: int = 8192,
    recipe: str = MX9_RECIPE,
) -> Trainer.Config:
    """Evaluate an HF causal LM with flavor ``full``.

    ``HF_MODEL_PATH`` names a local HF model directory: config, tokenizer and
    safetensors weights. The model config is built from it here, before CLI
    overrides apply, so it cannot be a CLI field. ``seq_len`` is the sequence
    length. ``recipe`` is an MX yaml, ``MX6_RECIPE`` or ``MX9_RECIPE``; an empty
    string leaves the model in BF16.
    """
    model_path = os.environ.get("HF_MODEL_PATH", "")
    if not model_path:
        raise ValueError("Set HF_MODEL_PATH to a local HF model directory")
    model_config = build_model_config("full", seq_len=seq_len, hf_model=model_path)
    config = Trainer.Config(
        hf_assets_path=model_path,
        model=model_config,
        tokenizer=HFBackendTokenizer.Config(),
        loss=CrossEntropyLoss.Config(global_vocab_size=model_config.vocab_size),
        optim=Optim.Config(optimizer=OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=r".*", lr=0.0)])),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            max_context_length=seq_len,
            steps=1 if recipe else 0,
        ),
        dataloader=GrainDataLoader.Config(dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"])),
        validator=Validator.Config(
            freq=10,
            steps=10,
            dataloader=GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["wikitext_test"]),
                shuffle=False,
            ),
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpointer=CheckpointManager.Config(
            initial_load_path=model_path,
            initial_load_in_hf=True,
            load_only=True,
        ),
        activation_remat=None,
        profiler=Profiler.Config(enable_profiling=False),
        debug=DebugConfig(seed=1234),
        dump_folder="outputs/hf_backend",
    )
    if recipe:
        config.model_converters = ModelConvertersContainer.Config(converters=[
            ModelOptConverter.Config(recipe=recipe),
        ])
    return config


def hf_backend_mx9_wa() -> Trainer.Config:
    return hf_backend(recipe=MX9_RECIPE)


def hf_backend_mx6_wa() -> Trainer.Config:
    return hf_backend(recipe=MX6_RECIPE)


def hf_backend_bf16() -> Trainer.Config:
    return hf_backend(recipe="")
