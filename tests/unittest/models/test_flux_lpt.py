# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import torch

from alto.components.converter import ModelOptConverter
from alto.models.flux.config_registry import flux_debugmodel, flux_debugmodel_lpt
from alto.models.flux.trainer import FluxTrainer
from alto.nn.lpt import LPTModule
from alto.train import Trainer, trainer_for
from torchtitan.models.flux.trainer import FluxTrainer as TitanFluxTrainer

_ASSETS = (
    Path(__file__).resolve().parents[3]
    / "3rdparty"
    / "torchtitan"
    / "tests"
    / "assets"
)


def _use_offline_test_assets(config):
    """Point a registry config at local encoders with random weights."""
    config.hf_assets_path = str(_ASSETS / "tokenizer")
    config.tokenizer.test_mode = True
    config.tokenizer.t5_tokenizer_path = config.hf_assets_path
    config.tokenizer.clip_tokenizer_path = config.hf_assets_path
    config.encoder.random_init = True
    config.encoder.clip_encoder = str(
        _ASSETS / "flux_test_encoders" / "clip-vit-large-patch14"
    )
    config.encoder.t5_encoder = str(_ASSETS / "flux_test_encoders" / "t5-v1_1-xxl")
    return config


def test_flux_debugmodel_lpt_selects_combined_trainer():
    config = _use_offline_test_assets(flux_debugmodel_lpt())
    assert isinstance(config, TitanFluxTrainer.Config)
    assert type(config) is TitanFluxTrainer.Config
    assert trainer_for(config) is FluxTrainer
    assert trainer_for(object()) is Trainer
    assert FluxTrainer.__mro__[1] is TitanFluxTrainer
    assert FluxTrainer.__mro__[2] is Trainer
    assert FluxTrainer.train_step is Trainer.train_step
    assert config.tokenizer.test_mode is True
    assert config.encoder.random_init is True
    assert Path(config.encoder.t5_encoder).is_dir()
    assert Path(config.encoder.clip_encoder).is_dir()
    assert Path(config.dataloader.dataset.source.path).is_dir()
    assert Path(config.hf_assets_path).is_dir()
    bf16 = flux_debugmodel()
    tokens = 32 * bf16.training.max_context_length
    assert bf16.training.num_tokens_per_microbatch_per_dp_rank == tokens
    assert config.training.num_tokens_per_microbatch_per_dp_rank == tokens


def test_flux_recipe_installs_mxfp4_linears():
    config = _use_offline_test_assets(flux_debugmodel_lpt())
    converter = config.model_converters.converters[0]
    assert isinstance(converter, ModelOptConverter.Config)
    ModelOptConverter(
        converter,
        parallelism_context=None,
        model_compile_enabled=False,
    ).convert_config(config.model)

    block = config.model.double_blocks[0]
    assert type(block.img_mlp_in).__qualname__ == "LPTLinear.Config"
    assert block.img_mlp_in.training_op_config.precision == "mxfp4"
    assert block.img_mlp_in.inplace_wgrad_accum is False
    assert type(block.inner_attention).__qualname__ == "VarlenInnerAttention.Config"
    assert (
        config.training.num_tokens_per_microbatch_per_dp_rank
        == 32 * config.training.max_context_length
    )
    assert type(config.model.final_layer_config.linear).__qualname__ == "Linear.Config"
    assert type(config.model.final_layer_config.adaln_linear).__qualname__ == "Linear.Config"
    assert type(config.model.time_in_config.in_layer).__qualname__ == "LPTLinear.Config"
    assert type(config.model.vector_in_config.in_layer).__qualname__ == "Linear.Config"
    assert type(config.model.vector_in_config.out_layer).__qualname__ == "Linear.Config"
    assert type(block.img_mod.lin).__qualname__ == "Linear.Config"
    assert type(block.txt_mod.lin).__qualname__ == "Linear.Config"
    single = config.model.single_blocks[0]
    assert type(single.modulation.lin).__qualname__ == "Linear.Config"
    assert type(single.linear1).__qualname__ == "LPTLinear.Config"
    assert type(single.inner_attention).__qualname__ == "VarlenInnerAttention.Config"

    with torch.device("meta"):
        model = config.model.build()
    assert not isinstance(model.img_in, LPTModule)
    assert not isinstance(model.txt_in, LPTModule)
    assert isinstance(model.double_blocks[0].img_mlp[0], LPTModule)
    assert type(model.double_blocks[0].inner_attention).__name__ == "VarlenInnerAttention"
