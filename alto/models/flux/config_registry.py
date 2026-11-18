# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import re
from dataclasses import replace
from pathlib import Path

import torch

from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.optim import AdamW, DistMuon, MPAdam, OptimizersContainer
from torchtitan.distributed.flex_shard import BucketConfig, ComputeLayout, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName
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
_LR = 2e-4


def _install_dist_muon(config: FluxTrainer.Config) -> None:
    """Put every matrix on DistMuon and leave vectors on AdamW.

    One GPU still stores parameters as DTensors on the size-1 ``dp_shard``
    axis, so the compute layout owns each whole matrix on that axis.
    ``match_rms_adamw`` keeps the configured learning rate on the AdamW scale.
    """
    module = config.model.build()
    matrix_names = tuple(
        name for name, parameter in module.named_parameters() if parameter.ndim >= 2
    )
    if not matrix_names:
        raise RuntimeError("Flux has no matrix parameters for DistMuon")
    layout = ComputeLayout(
        shardings_by_mesh_axis={MeshAxisName.DP_SHARD.value: Owned()},
        compute_dtype=torch.bfloat16,
    )
    config.optim.optimizer = OptimizersContainer.Config(
        optimizers=[
            DistMuon.Config(
                pattern="^(?:%s)$" % "|".join(re.escape(name) for name in matrix_names),
                bucket_configs=(BucketConfig(name="matrices", patterns=("*",)),),
                compute_sharding_by_fqn={name: layout for name in matrix_names},
                lr=_LR,
                weight_decay=0.1,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(
                pattern=r".*",
                lr=_LR,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=0.1,
            ),
        ]
    )


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
    _install_dist_muon(config)
    config.optim.lr_scheduler.warmup_steps = 16
    config.optim.lr_scheduler.decay_ratio = 0.0
    config.optim.max_norm = 2.0
    config.training.steps = 100
    config.dump_folder = "flux_debugmodel-outputs"
    return config


def flux_debugmodel_lpt() -> FluxTrainer.Config:
    """Flux debug model whose Linear layers, including conditioning, run in MXFP4.

    Matrix weights use the same DistMuon schedule as the BF16 debug run.
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
