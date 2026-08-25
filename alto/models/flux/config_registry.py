# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

from torchtitan.models.flux.config_registry import (
    flux_debugmodel as flux_debugmodel_orig,
    flux_dev as flux_dev_orig,
    flux_schnell as flux_schnell_orig,
)
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan.protocols.model_converter import ModelConvertersContainer

from alto.components.converter import ModelOptConverter

__all__ = [
    "flux_debugmodel",
    "flux_debugmodel_lpt",
    "flux_debugmodel_lpt_mse_4_6_shifted",
    "flux_debugmodel_lpt_mse_4_6_strict",
    "flux_debugmodel_lpt_precision_schedule",
    "flux_debugmodel_mxfp8",
    "flux_debugmodel_nvfp4",
    "flux_dev",
    "flux_dev_lpt",
    "flux_dev_lpt_mse_4_6_shifted",
    "flux_dev_lpt_mse_4_6_strict",
    "flux_dev_lpt_precision_schedule",
    "flux_dev_mxfp8",
    "flux_dev_nvfp4",
    "flux_schnell",
    "flux_schnell_lpt",
    "flux_schnell_lpt_mse_4_6_shifted",
    "flux_schnell_lpt_mse_4_6_shifted_mlperf",
    "flux_schnell_lpt_mse_4_6_shifted_forward_bf16",
    "flux_schnell_lpt_mse_4_6_shifted_backward_bf16",
    "flux_schnell_lpt_mse_4_6_strict",
    "flux_schnell_lpt_precision_schedule",
    "flux_schnell_lpt_distill_off_policy",
    "flux_schnell_lpt_distill_on_policy",
    "flux_schnell_mxfp8",
    "flux_schnell_mlperf_mxfp8",
    "flux_schnell_nvfp4",
    "flux_schnell_lpt_mse_4_6_shifted_schedule_mlperf",
]


FLUX_LPT_RECIPE = str(Path(__file__).with_name("configs") / "lpt_recipe.yaml")
FLUX_LPT_MSE_4_6_SHIFTED_RECIPE = str(Path(__file__).with_name("configs") / "lpt_mse_4_6_shifted_recipe.yaml")
FLUX_LPT_MSE_4_6_SHIFTED_FORWARD_BF16_RECIPE = str(
    Path(__file__).with_name("configs") / "lpt_mse_4_6_shifted_forward_bf16_recipe.yaml"
)
FLUX_LPT_MSE_4_6_SHIFTED_BACKWARD_BF16_RECIPE = str(
    Path(__file__).with_name("configs") / "lpt_mse_4_6_shifted_backward_bf16_recipe.yaml"
)
FLUX_LPT_MSE_4_6_STRICT_RECIPE = str(Path(__file__).with_name("configs") / "lpt_mse_4_6_strict_recipe.yaml")
FLUX_LPT_PRECISION_SCHEDULE_RECIPE = str(Path(__file__).with_name("configs") / "lpt_precision_schedule_recipe.yaml")
FLUX_DISTILL_LPT_RECIPE = str(Path(__file__).with_name("configs") / "distill_lpt_recipe.yaml")
FLUX_MXFP8_RECIPE = str(Path(__file__).with_name("configs") / "mxfp8_recipe.yaml")
FLUX_NVFP4_RECIPE = str(Path(__file__).with_name("configs") / "nvfp4_recipe.yaml")


def _with_recipe(config: FluxTrainer.Config, recipe: str) -> FluxTrainer.Config:
    config.activation_checkpoint.mode = "none"
    config.model_converters = ModelConvertersContainer.Config(
        converters=[
            ModelOptConverter.Config(recipe=recipe),
        ],
    )
    return config


def _with_lpt(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_RECIPE)


def _with_lpt_mse_4_6_shifted(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_MSE_4_6_SHIFTED_RECIPE)


def _with_lpt_mse_4_6_shifted_forward_bf16(
    config: FluxTrainer.Config,
) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_MSE_4_6_SHIFTED_FORWARD_BF16_RECIPE)


def _with_lpt_mse_4_6_shifted_backward_bf16(
    config: FluxTrainer.Config,
) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_MSE_4_6_SHIFTED_BACKWARD_BF16_RECIPE)


def _with_lpt_mse_4_6_strict(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_MSE_4_6_STRICT_RECIPE)


def _with_lpt_precision_schedule(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_LPT_PRECISION_SCHEDULE_RECIPE)


def _with_lpt_distillation(config: FluxTrainer.Config, mode: str) -> FluxTrainer.Config:
    config = _with_recipe(config, FLUX_DISTILL_LPT_RECIPE)
    config.distillation.enable = True
    config.distillation.mode = mode
    config.distillation.num_rollout_steps = 4

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.5e-5
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.lr_scheduler.warmup_steps = 1000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.max_norm = 1.0
    config.training.local_batch_size = 32
    config.training.steps = 10_000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 5000
    # MLPerf COCO validation is a finite, fixed 29,696-sample dataset.
    # Consume it to exhaustion so changing the local batch size does not
    # silently shorten validation.
    config.validator.steps = -1
    config.validator.save_img_count = 0
    config.validator.validation_timeout_seconds = 900
    config.validator.dataloader.dataset = "mlperf-coco-validation"
    config.validator.dataloader.dataset_path = "/dataset/coco"
    config.validator.dataloader.mlperf_manifest_path = "/dataset/val2014_30k.tsv"
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    config.debug.seed = 10556

    return config


def _with_mxfp8(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_MXFP8_RECIPE)


def _with_nvfp4(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_NVFP4_RECIPE)


def flux_debugmodel() -> FluxTrainer.Config:
    return flux_debugmodel_orig()


def flux_debugmodel_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_debugmodel())


def flux_debugmodel_lpt_mse_4_6_shifted() -> FluxTrainer.Config:
    return _with_lpt_mse_4_6_shifted(flux_debugmodel())


def flux_debugmodel_lpt_mse_4_6_strict() -> FluxTrainer.Config:
    return _with_lpt_mse_4_6_strict(flux_debugmodel())


def flux_debugmodel_lpt_precision_schedule() -> FluxTrainer.Config:
    return _with_lpt_precision_schedule(flux_debugmodel())


def flux_debugmodel_mxfp8() -> FluxTrainer.Config:
    return _with_mxfp8(flux_debugmodel())


def flux_debugmodel_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_debugmodel())


def flux_dev() -> FluxTrainer.Config:
    return flux_dev_orig()


def flux_dev_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_dev())


def flux_dev_lpt_mse_4_6_shifted() -> FluxTrainer.Config:
    return _with_lpt_mse_4_6_shifted(flux_dev())


def flux_dev_lpt_mse_4_6_strict() -> FluxTrainer.Config:
    return _with_lpt_mse_4_6_strict(flux_dev())


def flux_dev_lpt_precision_schedule() -> FluxTrainer.Config:
    return _with_lpt_precision_schedule(flux_dev())


def flux_dev_mxfp8() -> FluxTrainer.Config:
    return _with_mxfp8(flux_dev())


def flux_dev_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_dev())


def flux_schnell() -> FluxTrainer.Config:
    return flux_schnell_orig()


def flux_schnell_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_schnell())


def flux_schnell_lpt_mse_4_6_shifted() -> FluxTrainer.Config:
    config = _with_lpt_mse_4_6_shifted(flux_schnell())

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.5e-4
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.lr_scheduler.warmup_steps = 2000
    # config.lr_scheduler.warmup_steps = 0
    config.lr_scheduler.decay_ratio = 0.0
    config.training.max_norm = 1.0
    config.training.local_batch_size = 32
    # config.training.steps = 60000
    config.training.steps = 45000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 1024 # NGPU =8 
    config.validator.steps = -1
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    config.debug.seed = 10556

    config.validator.dataloader.dataset = "mlperf-coco-validation"
    config.validator.dataloader.dataset_path = "/dataset/coco"
    config.validator.dataloader.mlperf_manifest_path = "/dataset/val2014_30k.tsv"


    return config


def flux_schnell_lpt_mse_4_6_shifted_mlperf() -> FluxTrainer.Config:
    config = _with_lpt_mse_4_6_shifted(flux_schnell())
    # Match NVIDIA MLPerf FLUX sample cadence with four GPUs and local batch 32.
    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.5e-4
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.lr_scheduler.warmup_steps = 800
    config.lr_scheduler.decay_ratio = 0.0
    config.training.max_norm = 1.0
    config.training.local_batch_size = 32
    config.training.steps = 69_632
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 2_048
    config.validator.steps = 232
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    config.debug.seed = 10556
    return config


def flux_schnell_lpt_mse_4_6_shifted_forward_bf16() -> FluxTrainer.Config:
    config = _with_lpt_mse_4_6_shifted_forward_bf16(flux_schnell())

    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.5e-4
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.max_norm = 1.0
    config.training.local_batch_size = 32
    config.training.steps = 50000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 1024 # NGPU =8 
    config.validator.steps = -1
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    config.debug.seed = 10556
    
    return config


def flux_schnell_lpt_mse_4_6_shifted_backward_bf16() -> FluxTrainer.Config:
    config = _with_lpt_mse_4_6_shifted_backward_bf16(flux_schnell())
    config.optimizer.lr = 2.5e-4
    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.steps = 30_000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    return config


def flux_schnell_lpt_mse_4_6_strict() -> FluxTrainer.Config:
    config = _with_lpt_mse_4_6_strict(flux_schnell())
    config.optimizer.lr = 2.5e-4
    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.steps = 30_000
    config.dataloader.classifier_free_guidance_prob = 0.1
    # config.validator.enable = True
    # config.validator.freq = 5461
    # config.validator.steps = 619
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    return config


def flux_schnell_lpt_precision_schedule() -> FluxTrainer.Config:
    config = _with_lpt_precision_schedule(flux_schnell())
    config.optimizer.lr = 2.5e-4
    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.steps = 30_000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    return config


def flux_schnell_lpt_distill_off_policy() -> FluxTrainer.Config:
    config = _with_lpt_distillation(flux_schnell(), "off_policy")
    return config


def flux_schnell_lpt_distill_on_policy() -> FluxTrainer.Config:
    config = flux_schnell_lpt_distill_off_policy()
    config.distillation.mode = "on_policy"
    return config


def flux_schnell_mxfp8() -> FluxTrainer.Config:
    return _with_mxfp8(flux_schnell())


def flux_schnell_mlperf_mxfp8() -> FluxTrainer.Config:
    config = flux_schnell_mxfp8()
    config.optimizer.lr = 2.5e-4
    config.lr_scheduler.warmup_steps = 3000
    config.lr_scheduler.decay_ratio = 0.0
    config.training.steps = 30_000
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 5461
    config.validator.steps = 619
    config.validator.save_img_count = 0
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    return config

def flux_schnell_lpt_mse_4_6_shifted_schedule_mlperf() -> FluxTrainer.Config:
    config =  _with_lpt_precision_schedule(flux_schnell())
    # Match NVIDIA MLPerf FLUX sample counts with 4 GPUs and local batch size 32.
    config.optimizer.name = "AdamW"
    config.optimizer.lr = 2.5e-4
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.95
    config.optimizer.eps = 1e-8
    config.optimizer.weight_decay = 0.1
    config.lr_scheduler.warmup_steps = 8704
    config.lr_scheduler.decay_ratio = 0.0
    config.training.max_norm = 1.0
    config.training.local_batch_size = 32
    config.training.steps = 69_632
    config.dataloader.classifier_free_guidance_prob = 0.1
    config.validator.enable = True
    config.validator.freq = 34_816
    config.validator.steps = 232
    config.validator.save_img_count = 0
    config.validator.dataloader.dataset = "mlperf-coco-validation"
    config.validator.dataloader.dataset_path = "/dataset/coco"
    config.validator.dataloader.mlperf_manifest_path = "/dataset/val2014_30k.tsv"
    config.validator.dataloader.classifier_free_guidance_prob = 0.0
    config.validation.enable_classifier_free_guidance = False
    config.validation.denoising_steps = 4
    config.debug.seed = 10556
    return config

def flux_schnell_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_schnell())
