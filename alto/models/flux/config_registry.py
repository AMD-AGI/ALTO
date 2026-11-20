# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import replace
from pathlib import Path

from torchtitan.components.data import GrainDataLoader
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.optim import MPAdam
from torchtitan.models.flux.data import (
    DATASETS,
    FluxCollator,
    FluxSampleProcessor,
    FluxValidationDatasetConfig,
)
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.models.flux.utils import (
    IMAGE_LATENT_SIZE_RATIO,
    PATCH_HEIGHT,
    PATCH_WIDTH,
)
from torchtitan.models.flux.validator import FluxValidator, SamplingConfig
from torchtitan_recipes.models.flux import flux_schnell as flux_schnell_orig
from torchtitan_recipes.tests.models.flux import (
    flux_debugmodel as flux_debugmodel_orig,
)

from alto.components.converter import ModelOptConverter
from alto.models.batch import set_batch_sizes

__all__ = [
    "flux_debugmodel",
    "flux_debugmodel_lpt",
    "flux_schnell",
    "flux_schnell_lpt",
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


# MLPerf flux_schnell: 16 images, 256px, T5 length 256.
# Image tokens are (256 / 8 / 2)^2 = 256, plus 256 text tokens.
_SCHNELL_IMG_SIZE = 256
_SCHNELL_T5_LEN = 256
_SCHNELL_LOCAL_BATCH = 16
# MLPerf batch_size is per rank. run.sh defaults to 8 GPUs, all data parallel.
# Two microbatches are accumulated before each optimizer step, so one step
# sees 16 * 8 * 2 = 256 images. 262,144 training images is then 1,024
# optimizer steps. Validation does not accumulate: 29,696 / (16 * 8) = 232.
_SCHNELL_DP_DEGREE = 8
_SCHNELL_GRAD_ACCUM = 2
_EVAL_EVERY_N_SAMPLES = 262_144
_EVAL_NUM_SAMPLES = 29_696


def _schnell_seq_len() -> int:
    latent = _SCHNELL_IMG_SIZE // IMAGE_LATENT_SIZE_RATIO
    image_tokens = (latent // PATCH_WIDTH) * (latent // PATCH_HEIGHT)
    return image_tokens + _SCHNELL_T5_LEN


def _apply_schnell_runtime(config: FluxTrainer.Config) -> FluxTrainer.Config:
    """Point encoders at local assets and apply the MLPerf schnell schedule.

    Training still streams ``pixparse/cc12m-wds`` and encodes each batch with
    the autoencoder, T5, and CLIP. The old MLPerf job read a precomputed
    parquet cache; this tree has no loader for that cache. Image resize, crop,
    RGB conversion, and the ``[-1, 1]`` scale are the online preprocess.
    """
    config.hf_assets_path = str(_ASSETS / "tokenizer")
    config.tokenizer.test_mode = False
    config.tokenizer.t5_tokenizer_path = "google/t5-v1_1-xxl"
    config.tokenizer.clip_tokenizer_path = "openai/clip-vit-large-patch14"
    config.tokenizer.max_t5_encoding_len = _SCHNELL_T5_LEN
    config.encoder.random_init = False
    config.encoder.t5_encoder = "google/t5-v1_1-xxl"
    config.encoder.clip_encoder = "openai/clip-vit-large-patch14"
    config.encoder.autoencoder_path = str(
        _TORCHTITAN / "assets" / "hf" / "FLUX.1-dev" / "ae.safetensors"
    )

    dataset = config.dataloader.dataset
    processor = dataset.processor
    if not isinstance(processor, FluxSampleProcessor.Config):
        raise TypeError("Flux schnell dataset must use FluxSampleProcessor")
    config.dataloader.dataset = replace(
        dataset,
        processor=replace(
            processor,
            img_size=_SCHNELL_IMG_SIZE,
            prompt_dropout_prob=0.1,
        ),
    )

    seq_len = _schnell_seq_len()
    config.training.max_context_length = seq_len
    images_per_step = (
        _SCHNELL_LOCAL_BATCH * _SCHNELL_DP_DEGREE * _SCHNELL_GRAD_ACCUM
    )
    set_batch_sizes(
        config.training,
        seq_len=seq_len,
        local_batch_size=_SCHNELL_LOCAL_BATCH,
        global_batch_size=images_per_step,
    )
    config.training.steps = 30_000
    config.training.disable_accelerator_graphs = True

    adam = config.optim.optimizer.optimizers[0]
    adam.lr = 2e-4
    adam.eps = 1e-8
    config.optim.lr_scheduler.warmup_steps = 0
    config.optim.lr_scheduler.decay_ratio = 0.0
    config.optim.max_norm = 1.0

    config.metrics.log_freq = 1
    config.metrics.disable_color_printing = True

    coco = DATASETS["coco-validation"]
    coco_processor = coco.processor
    if not isinstance(coco_processor, FluxSampleProcessor.Config):
        raise TypeError("COCO validation must use FluxSampleProcessor")
    validation = replace(
        coco,
        processor=replace(coco_processor, img_size=_SCHNELL_IMG_SIZE),
    )
    images_per_val_step = _SCHNELL_LOCAL_BATCH * _SCHNELL_DP_DEGREE
    if (
        _EVAL_EVERY_N_SAMPLES % images_per_step
        or _EVAL_NUM_SAMPLES % images_per_val_step
    ):
        raise ValueError(
            f"{images_per_step} images/step must divide {_EVAL_EVERY_N_SAMPLES}, "
            f"and {images_per_val_step} images/val step must divide {_EVAL_NUM_SAMPLES}"
        )
    config.validator = FluxValidator.Config(
        freq=_EVAL_EVERY_N_SAMPLES // images_per_step,
        steps=_EVAL_NUM_SAMPLES // images_per_val_step,
        sampling=SamplingConfig(
            enable_classifier_free_guidance=True,
            classifier_free_guidance_scale=5.0,
            denoising_steps=50,
        ),
        dataloader=GrainDataLoader.Config(
            dataset=FluxValidationDatasetConfig(dataset=validation),
            collator=FluxCollator.Config(),
            streaming_shuffle_buffer_size=128,
        ),
        save_img_count=0,
        save_img_folder="img",
    )
    return config


def flux_schnell() -> FluxTrainer.Config:
    """FLUX.1-schnell with the MLPerf training schedule.

    Each rank trains 16 images at 256px. With 8 data-parallel ranks and
    gradient accumulation 2, one optimizer step sees 256 images. Evaluation
    runs every 1,024 steps (262,144 images) and each pass is 232 steps
    (29,696 images). Training runs for 30,000 optimizer steps.
    The reference eval loss is 0.586; this trainer does not stop on it.
    """
    config = flux_schnell_orig()
    config = _apply_schnell_runtime(config)
    config.dump_folder = "flux_schnell-outputs"
    return config


def flux_schnell_lpt() -> FluxTrainer.Config:
    """FLUX.1-schnell on the MLPerf schedule, with MXFP4 linear layers."""
    config = flux_schnell()
    config.dump_folder = "flux_schnell_lpt-outputs"
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(
                recipe="./alto/models/flux/configs/lpt_recipe.yaml",
            )
        ],
    )
    return config
