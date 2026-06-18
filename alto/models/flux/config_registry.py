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
    "flux_debugmodel_nvfp4",
    "flux_dev",
    "flux_dev_lpt",
    "flux_dev_nvfp4",
    "flux_schnell",
    "flux_schnell_lpt",
    "flux_schnell_nvfp4",
]


FLUX_LPT_RECIPE = str(Path(__file__).with_name("configs") / "lpt_recipe.yaml")
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


def _with_nvfp4(config: FluxTrainer.Config) -> FluxTrainer.Config:
    return _with_recipe(config, FLUX_NVFP4_RECIPE)


def flux_debugmodel() -> FluxTrainer.Config:
    return flux_debugmodel_orig()


def flux_debugmodel_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_debugmodel())


def flux_debugmodel_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_debugmodel())


def flux_dev() -> FluxTrainer.Config:
    return flux_dev_orig()


def flux_dev_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_dev())


def flux_dev_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_dev())


def flux_schnell() -> FluxTrainer.Config:
    return flux_schnell_orig()


def flux_schnell_lpt() -> FluxTrainer.Config:
    return _with_lpt(flux_schnell())


def flux_schnell_nvfp4() -> FluxTrainer.Config:
    return _with_nvfp4(flux_schnell())
