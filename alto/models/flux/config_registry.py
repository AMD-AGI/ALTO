# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import replace
from pathlib import Path

from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.optim import MPAdam
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan_recipes.tests.models.flux import (
    flux_debugmodel as flux_debugmodel_orig,
)

from alto.components.converter import ModelOptConverter
from alto.models.batch import set_batch_sizes

__all__ = [
    "flux_debugmodel",
    "flux_debugmodel_lpt",
]

# Upstream recipe paths are relative to the torchtitan root. ALTO launches
# with this repository as the working directory.
_TORCHTITAN = Path(__file__).resolve().parents[3] / "3rdparty" / "torchtitan"
_ASSETS = _TORCHTITAN / "tests" / "assets"


def flux_debugmodel() -> FluxTrainer.Config:
    """Flux debug model with pretrained T5, CLIP, and the FLUX autoencoder.

    The returned object stays ``FluxTrainer.Config``. The transformer itself
    is still the random debug model. ``ae.safetensors`` is the local file
    produced by ``scripts/download_hf_assets.py``.
    """
    config = flux_debugmodel_orig()
    config.hf_assets_path = str(_ASSETS / "tokenizer")
    config.tokenizer.test_mode = False
    config.tokenizer.t5_tokenizer_path = "google/t5-v1_1-xxl"
    config.tokenizer.clip_tokenizer_path = "openai/clip-vit-large-patch14"
    config.encoder.random_init = False
    config.encoder.t5_encoder = "google/t5-v1_1-xxl"
    config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
    config.encoder.autoencoder_path = str(
        _TORCHTITAN / "assets" / "hf" / "FLUX.1-dev" / "ae.safetensors"
    )
    dataset = config.dataloader.dataset
    config.dataloader.dataset = replace(
        dataset,
        source=replace(dataset.source, path=str(_ASSETS / "cc12m_test")),
    )
    config.profiler.enable_profiling = False
    config.metrics.enable_tensorboard = True
    config.metrics.log_weight_update_distance = True
    config.debug.seed = 1234
    # Shared with the MXFP4 recipe. Conditioning linears see one row per
    # sample, so the microbatch must match the MXFP4 group of 32.
    set_batch_sizes(
        config.training,
        seq_len=config.training.max_context_length,
        local_batch_size=32,
    )
    config.optim.optimizer.optimizers[0].lr = 2e-4
    config.optim.lr_scheduler.warmup_steps = 16
    config.optim.lr_scheduler.decay_ratio = 0.0
    config.optim.max_norm = 2.0
    config.training.steps = 100
    config.dump_folder = "flux_debugmodel-outputs"
    return config


def flux_debugmodel_lpt() -> FluxTrainer.Config:
    """Flux debug model whose Linear layers, including conditioning, run in MXFP4.

    Weights are updated with MPAdam. The mantissa learning rate and weight
    decay stay on the shared debug schedule. The exponent step is
    ``lr * lr_e_ratio``.
    """
    config = flux_debugmodel()
    config.dump_folder = "flux_debugmodel_lpt-outputs"
    # adam = config.optim.optimizer.optimizers[0]
    # config.optim.optimizer.optimizers = [
    #     MPAdam.Config(
    #         pattern=adam.pattern,
    #         lr=adam.lr,
    #         weight_decay=adam.weight_decay,
    #         weight_decay_e=adam.weight_decay,
    #         lr_e_ratio=20.0,
    #     )
    # ]
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(
                recipe="./alto/models/flux/configs/lpt_recipe.yaml",
            )
        ],
    )
    return config
