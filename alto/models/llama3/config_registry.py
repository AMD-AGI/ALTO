# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Llama 3 configs.

Upstream recipes supply the model and optimizer skeleton. These functions put
back the ALTO dataset paths, batch sizes, schedules, checkpoints, and modifier
recipes.
"""

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.megatron import megatron_dataset_config
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.model_converter import ModelConvertersContainer
from torchtitan.components.validator import Validator
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import Attention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.data import c4_document_capacity
from torchtitan_recipes.models.llama3 import instella_3b as _instella_3b
from torchtitan_recipes.models.llama3 import llama3_1b as _llama3_1b
from torchtitan_recipes.models.llama3 import llama3_8b as _llama3_8b
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel as _llama3_debugmodel

from alto.components.converter import ModelOptConverter
from alto.models.patcher import ModelPatcher

__all__ = [
    "llama3_debugmodel",
    "llama3_debugmodel_opt",
    "llama3_debugmodel_lpt",
    "llama3_1b",
    "llama3_1b_opt",
    "llama3_1b_lpt",
    "llama3_1b_gptq",
    "llama3_1b_awq",
    "llama3_1b_mx9_wa",
    "llama3_1b_mx6_wa",
    "llama3_8b",
    "llama3_8b_pretrain",
    "llama3_8b_opt",
    "llama3_8b_lpt",
    "llama3_8b_gptq",
    "llama3_8b_rtn",
    "llama3_8b_awq",
    "llama3_8b_wanda",
    "llama3_8b_sparsegpt",
    "llama3_8b_magnitude",
    "llama3_8b_admm",
    "llama3_8b_alps",
    "llama3_8b_wanda_structured",
    "llama3_8b_obs",
    "llama3_8b_admm_structured",
    "llama3_8b_cosine_similarity",
    "instella_3b",
    "instella_3b_opt",
    "instella_3b_lpt",
]

_CONFIGS = "./alto/models/llama3/configs"
LLAMA3_1B_PATH = (
    "/group/archive_dataset_6_nobkup/archive_modelzoo/sequence_learning/weights/"
    "nlp-pretrained-model/meta-llama/Llama-3.2-1B"
)
LLAMA3_8B_PATH = "/workspace/Model-Optimizer/models/meta-llama/Llama-3.1-8B"
LLAMA3_8B_PRETRAIN_PATH = (
    "/huggingface/hub/models--unsloth--Llama-3.1-8B/snapshots/"
    "3f0d51f8e5640f98f1a96ea9044a0e55c0a83814"
)
INSTELLA_3B_PATH = (
    "/group/ossmodelzoo/hanwang2/huggingface/hub/models--amd--Instella-3B-Stage1/"
    "snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e"
)
C4_TRAIN_IDX = "/workspace/workspace/megatron_dataset/data/c4-train.en_6_text_document.idx"
C4_VAL_IDX = (
    "/workspace/workspace/megatron_dataset/data/"
    "c4-validation-91205-samples.en_text_document.idx"
)


def _source(name: str):
    if name.endswith(".idx"):
        return megatron_dataset_config(name)
    return DATASETS[name]


def _with_sdpa(config: Trainer.Config) -> Trainer.Config:
    """Use SDPA for every Llama attention layer.

    ALTO's low-precision attention implementation subclasses SDPA, and SDPA is
    also the intended fixed-shape backend for all Llama recipes in this registry.
    """
    for _fqn, attention, _parent, _attr in config.model.traverse(Attention.Config):
        attention.inner_attention = get_attention_config("sdpa")
    ModelPatcher.patch_model_config(config.model)
    return config


def _dataloader(name: str, *, micro_tokens: int, shuffle: bool, bound_documents: bool):
    return GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(dataset=_source(name)),
        shuffle=shuffle,
        max_num_documents=(
            c4_document_capacity(micro_tokens) if bound_documents else None
        ),
    )


def _set_local_batch(config: Trainer.Config, local_batch: int) -> None:
    seq_len = config.training.max_context_length
    micro_tokens = local_batch * seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = micro_tokens
    if getattr(config.dataloader, "max_num_documents", None) is not None:
        config.dataloader.max_num_documents = c4_document_capacity(micro_tokens)


def _set_global_batch(config: Trainer.Config, global_batch: int) -> None:
    config.training.num_tokens_per_train_step = (
        global_batch * config.training.max_context_length
    )


def _set_dataset(
    config: Trainer.Config,
    name: str,
    *,
    shuffle: bool = True,
    bound_documents: bool | None = None,
) -> None:
    if bound_documents is None:
        bound_documents = getattr(config.dataloader, "max_num_documents", None) is not None
    config.dataloader = _dataloader(
        name,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
        shuffle=shuffle,
        bound_documents=bound_documents,
    )


def _validator(
    name: str,
    *,
    freq: int,
    steps: int,
    micro_tokens: int,
    bound_documents: bool = False,
) -> Validator.Config:
    return Validator.Config(
        freq=freq,
        steps=steps,
        dataloader=_dataloader(
            name,
            micro_tokens=micro_tokens,
            shuffle=False,
            bound_documents=bound_documents,
        ),
    )


def _checkpoint(
    path: str,
    *,
    interval: int,
    hf: bool = False,
    load_only: bool = False,
) -> CheckpointManager.Config:
    return CheckpointManager.Config(
        interval=interval,
        initial_load_path=path,
        initial_load_in_hf=hf,
        load_only=load_only,
    )


def _prepare(
    config: Trainer.Config,
    *,
    steps: int,
    local_batch: int,
    global_batch: int | None = None,
    log_freq: int | None = 1,
) -> Trainer.Config:
    config.profiler.enable_profiling = False
    config.debug.seed = 1234
    config.activation_remat = None
    config.training.steps = steps
    if log_freq is not None:
        config.metrics.log_freq = log_freq
    _set_local_batch(config, local_batch)
    if global_batch is not None:
        _set_global_batch(config, global_batch)
    return config


def _attach(
    config: Trainer.Config,
    recipe: str,
    *,
    steps: int | None = None,
    lr: float | None = None,
    global_batch: int | None = None,
) -> Trainer.Config:
    config.model_converters = ModelConvertersContainer.Config(
        converters=[ModelOptConverter.Config(recipe=f"{_CONFIGS}/{recipe}.yaml")]
    )
    config.training.disable_cuda_graphs = True
    if steps is not None:
        config.training.steps = steps
    if lr is not None:
        config.optim.optimizer.optimizers[0].lr = lr
    if global_batch is not None:
        _set_global_batch(config, global_batch)
    return config


def llama3_debugmodel() -> Trainer.Config:
    config = _with_sdpa(
        _llama3_debugmodel(
            seq_len=2048,
            parallelism=ParallelismConfig(),
            attn_backend="sdpa",
        )
    )
    return _prepare(config, steps=0, local_batch=4, global_batch=16, log_freq=None)


def llama3_debugmodel_opt() -> Trainer.Config:
    return _attach(llama3_debugmodel(), "recipe", steps=1)


def llama3_debugmodel_lpt() -> Trainer.Config:
    return _attach(llama3_debugmodel(), "lpt_recipe", steps=10)


def llama3_1b() -> Trainer.Config:
    config = _prepare(
        _with_sdpa(_llama3_1b(seq_len=8192)),
        steps=0,
        local_batch=1,
        global_batch=10,
    )
    config.hf_assets_path = LLAMA3_1B_PATH
    _set_dataset(config, "c4_test")
    config.checkpointer = _checkpoint(LLAMA3_1B_PATH, interval=10, hf=True)
    config.validator = _validator(
        "wikitext_test",
        freq=10,
        steps=10,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
    )
    return config


def llama3_1b_opt() -> Trainer.Config:
    return _attach(llama3_1b(), "recipe", steps=1, lr=0.0)


def llama3_1b_lpt() -> Trainer.Config:
    return _attach(llama3_1b(), "lpt_recipe", steps=1000)


def llama3_1b_gptq() -> Trainer.Config:
    return _attach(llama3_1b(), "gptq_recipe", steps=1, lr=0.0)


def llama3_1b_awq() -> Trainer.Config:
    return _attach(llama3_1b(), "awq_recipe", steps=1, lr=0.0)


def llama3_1b_mx9_wa() -> Trainer.Config:
    return _attach(llama3_1b(), "mx9_wa_recipe", steps=1, lr=0.0)


def llama3_1b_mx6_wa() -> Trainer.Config:
    return _attach(llama3_1b(), "mx6_wa_recipe", steps=1, lr=0.0)


def llama3_8b_pretrain() -> Trainer.Config:
    config = _prepare(
        _with_sdpa(_llama3_8b(seq_len=8192)),
        steps=5000,
        local_batch=2,
        global_batch=384,
    )
    config.hf_assets_path = LLAMA3_8B_PRETRAIN_PATH
    config.dump_folder = "llama3_8b-mi308-pretrain-subset-gbs384-lr1e-4-outputs"
    config.metrics.enable_tensorboard = True
    config.optim.optimizer.optimizers[0].lr = 1e-4
    config.optim.lr_scheduler.warmup_steps = 500
    config.optim.lr_scheduler.decay_ratio = 0.9
    config.optim.lr_scheduler.decay_type = "cosine"
    config.optim.lr_scheduler.min_lr_factor = 0.0
    _set_dataset(config, C4_TRAIN_IDX)
    config.checkpointer = _checkpoint(
        LLAMA3_8B_PRETRAIN_PATH, interval=10, load_only=True
    )
    config.validator = _validator(
        C4_VAL_IDX,
        freq=768,
        steps=64,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
        bound_documents=True,
    )
    return config


def llama3_8b() -> Trainer.Config:
    config = _prepare(
        _with_sdpa(_llama3_8b(seq_len=2048)),
        steps=0,
        local_batch=1,
        global_batch=8,
    )
    config.hf_assets_path = LLAMA3_8B_PATH
    _set_dataset(config, "c4_test")
    config.checkpointer = _checkpoint(LLAMA3_8B_PATH, interval=10, hf=True)
    config.validator = _validator(
        "wikitext_test",
        freq=10,
        steps=10,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
        bound_documents=True,
    )
    return config


def llama3_8b_opt() -> Trainer.Config:
    return _attach(llama3_8b(), "recipe", steps=1)


def llama3_8b_lpt() -> Trainer.Config:
    config = llama3_8b_pretrain()
    config.dump_folder = (
        "llama3_8b-mi308-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-gbs384-lr1e-4-outputs"
    )
    return _attach(config, "lpt_recipe")


def llama3_8b_gptq() -> Trainer.Config:
    return _attach(llama3_8b(), "gptq_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_rtn() -> Trainer.Config:
    return _attach(llama3_8b(), "rtn_recipe", steps=1, lr=0.0)


def llama3_8b_awq() -> Trainer.Config:
    return _attach(llama3_8b(), "awq_recipe", steps=1, lr=0.0)


def llama3_8b_wanda() -> Trainer.Config:
    return _attach(llama3_8b(), "wanda_recipe", steps=1, lr=0.0)


def llama3_8b_sparsegpt() -> Trainer.Config:
    return _attach(llama3_8b(), "sparsegpt_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_magnitude() -> Trainer.Config:
    return _attach(llama3_8b(), "magnitude_recipe", steps=1, lr=0.0)


def llama3_8b_admm() -> Trainer.Config:
    return _attach(llama3_8b(), "admm_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_alps() -> Trainer.Config:
    return _attach(llama3_8b(), "alps_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_wanda_structured() -> Trainer.Config:
    return _attach(llama3_8b(), "wanda_structured_recipe", steps=1, lr=0.0)


def llama3_8b_obs() -> Trainer.Config:
    return _attach(llama3_8b(), "obs_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_admm_structured() -> Trainer.Config:
    return _attach(llama3_8b(), "admm_structured_recipe", steps=1, lr=0.0, global_batch=128)


def llama3_8b_cosine_similarity() -> Trainer.Config:
    return _attach(llama3_8b(), "cosine_similarity_recipe", steps=1, lr=0.0)


def instella_3b() -> Trainer.Config:
    config = _prepare(
        _with_sdpa(_instella_3b()),
        steps=0,
        local_batch=1,
        global_batch=10,
    )
    config.hf_assets_path = INSTELLA_3B_PATH
    _set_dataset(config, "c4_test")
    config.checkpointer = _checkpoint(INSTELLA_3B_PATH, interval=10, hf=True)
    config.validator = _validator(
        "wikitext_test",
        freq=10,
        steps=10,
        micro_tokens=config.training.num_tokens_per_microbatch_per_dp_rank,
    )
    return config


def instella_3b_opt() -> Trainer.Config:
    return _attach(instella_3b(), "recipe", steps=1)


def instella_3b_lpt() -> Trainer.Config:
    return _attach(instella_3b(), "lpt_recipe", steps=1000)
