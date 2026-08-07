# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from torchtitan.trainer import Trainer
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.models.gpt_oss.config_registry import (
    gpt_oss_debugmodel as gpt_oss_debugmodel_orig,
    gpt_oss_20b as gpt_oss_20b_orig,
)

from alto.components.converter import ModelOptConverter
from alto.components.m_adam import MAdamOptimizersContainer

__all__ = [
    "gpt_oss_debugmodel",
    "gpt_oss_debugmodel_lpt",
    "gpt_oss_debugmodel_obs_lpt",
    "gpt_oss_debugmodel_obs_bf16",
    "gpt_oss_20b",
    "gpt_oss_20b_pretrain",
    "gpt_oss_20b_pretrain_c4",
    "gpt_oss_20b_lpt",
    "gpt_oss_20b_lpt_fresh",
    "gpt_oss_20b_lpt_1dw",
    "gpt_oss_20b_lpt_no2dw",
    "gpt_oss_20b_adahop",
    "gpt_oss_20b_adahop_hadamard",
    "gpt_oss_20b_pretrain_c4_megatron",
    "gpt_oss_20b_lpt_c4",
    "gpt_oss_20b_grad_clip_lpt",
    "gpt_oss_debugmodel_grad_clip_lpt",
    "gpt_oss_debugmodel_grad_clip_obs_lpt",
    "gpt_oss_debugmodel_grad_clip_lpt_no_fsdp",
    "gpt_oss_debugmodel_lpt_1gpu_ckpt",
    "gpt_oss_debugmodel_obs_lpt_no_fsdp",
    "gpt_oss_debugmodel_moe_pattern_obs",
    "gpt_oss_debugmodel_moe_pattern_obs_no_fsdp",
    "gpt_oss_20b_moe_pattern_obs",
    "gpt_oss_20b_lpt_midmax",
    "gpt_oss_20b_lpt_uos",
    "gpt_oss_20b_lpt_uos6",
    "gpt_oss_20b_lpt_lowrank",
    "gpt_oss_20b_lpt_deosc",
    "gpt_oss_20b_lpt_madam",
    "gpt_oss_20b_lpt_madam_stable",
    "gpt_oss_20b_mxfp4_base"
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


def gpt_oss_debugmodel_lpt() -> Trainer.Config:
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config


def gpt_oss_debugmodel_obs_lpt() -> Trainer.Config:
    """gpt_oss debugmodel + MXFP4 + DebugObserver. Produces a per-step
    tensor dump under the path configured in
    ``configs/debug_observer_lpt_recipe.yaml``."""
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/debug_observer_lpt_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_debugmodel_lpt_1gpu_ckpt() -> Trainer.Config:
    """Single-GPU (no FSDP/EP/TP) MXFP4 debugmodel training that writes full,
    resumable checkpoints. Pairs with debug_train_gpt_oss_1gpu_ckpt.sh: run this
    on a login-node GPU to produce checkpoints we later load for per-expert
    stats. Model quality is irrelevant — we only need loadable checkpoints."""
    config = gpt_oss_debugmodel_lpt()
    # Force all sharding degrees to 1 (login node can't do FSDP).
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.compile.enable = False
    # Full (resumable) checkpoints, keep them all so any step can be loaded later.
    config.checkpoint.enable = True
    config.checkpoint.interval = 50
    config.checkpoint.keep_latest_k = 0
    return config


def gpt_oss_debugmodel_obs_lpt_no_fsdp() -> Trainer.Config:
    """Single-GPU (no FSDP/EP/TP) MXFP4 + DebugObserver. Loads a checkpoint
    produced by gpt_oss_debugmodel_lpt_1gpu_ckpt, runs a few steps, and dumps
    per-expert inputs/weights/grads to ./outputs/debug_obs_lpt.pt. Pairs with
    debug_stats_gpt_oss_1gpu.sh, which sets --checkpoint.initial_load_path."""
    config = gpt_oss_debugmodel_obs_lpt()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.compile.enable = False
    # Load pretrained weights only; the stats script points initial_load_path at
    # a step-N checkpoint and clears any resumable state so the step counter
    # starts at 0 and the observer actually fires.
    config.checkpoint.enable = True
    return config


def gpt_oss_debugmodel_obs_bf16() -> Trainer.Config:
    """gpt_oss debugmodel in plain BF16 + DebugObserver (no LPT). Used to
    capture the reference dump that visualizer can diff against the
    quantized run."""
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/debug_observer_bf16_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_20b() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = "/hf_home/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.dump_folder = "gpt_oss_20b-outputs"
    config.profiling.enable_profiling = False
    config.training.steps = 0
    config.training.local_batch_size = 1
    config.training.seq_len = 8192
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "c4_test"
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = "/hf_home/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_in_hf_quantized = True
    config.checkpoint.interval = 100
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.freq = 10
    config.validator.steps = 10
    config.activation_checkpoint.mode = "none"
    config.debug.seed = 1234
    return config


def gpt_oss_20b_pretrain() -> Trainer.Config:
    config = gpt_oss_20b_orig()
    config.hf_assets_path = "/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/"
    config.dump_folder = "gpt_oss_20b-mi300-pretrain-subset-lr4e-4-outputs"
    config.profiling.enable_profiling = False
    config.training.steps = 1200000 # set by mlperf
    config.training.local_batch_size = 1
    config.training.global_batch_size = 16 # can be edited for mlperf submission
    config.training.seq_len = 8192 # set by mlperf
    config.optimizer.lr = 4e-4 # can be edited for mlperf
    config.optimizer.weight_decay = 0.1 # set by mlperf
    config.optimizer.beta1 = 0.9 # set by mlperf
    config.optimizer.beta2 = 0.95 # set by mlperf
    config.optimizer.eps = 1e-5 # set by mlperf
    config.lr_scheduler.min_lr_factor = 0.1 # set by mlperf
    config.lr_scheduler.warmup_steps = 128 # can be edited for mlperf submission
    config.lr_scheduler.decay_ratio = 1 - 128 / config.training.steps
    config.lr_scheduler.decay_type = "cosine"
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.dataloader.dataset = "megatron"
    config.dataloader.dataset_path = "/data/c4-train.en_6_text_document.idx"
    config.parallelism.expert_parallel_degree = 8
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.checkpoint.enable = True
    config.checkpoint.interval = 1000
    config.checkpoint.keep_latest_k = 2
    config.validator.enable = True
    config.validator.dataloader.dataset = "wikitext_test"
    config.validator.dataloader.dataset_path = ""
    config.validator.freq = 768
    config.validator.steps = 64
    config.activation_checkpoint.mode = "none"
    config.debug.seed = 1234
    return config

def gpt_oss_20b_pretrain_c4_megatron() -> Trainer.Config:
    """gpt_oss_20b_pretrain using HuggingFace C4 dataset (bf16 baseline)."""
    config = gpt_oss_20b_pretrain()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-bf16-c4-outputs"
    config.dataloader.dataset = "megatron"
    config.dataloader.dataset_path = "/data/c4-train.en_6_text_document.idx"
    config.validator.dataloader.dataset = "megatron"
    config.validator.dataloader.dataset_path = "/data/c4-validation-91205-samples.en_text_document.idx"
    config.checkpoint.enable = True
    config.checkpoint.initial_load_path = None      # fresh run: do NOT load any checkpoint
    config.checkpoint.initial_load_in_hf = False
    config.checkpoint.initial_load_in_hf_quantized = False
    config.checkpoint.interval = 1000               # Save at step interval
    config.checkpoint.last_save_model_only = False   # save full ckpt at final step (model+optim+dataloader) so training can resume
    return config

def gpt_oss_20b_adahop() -> Trainer.Config:
    config = gpt_oss_20b_pretrain()
    config.training.global_batch_size = 16
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 8
    config.training.local_batch_size = 1
    config.activation_checkpoint.mode = "none"
    config.dataloader.dataset = "c4"
    config.dataloader.dataset_path = None
    config.validator.dataloader.dataset = "c4_validation"
    config.validator.dataloader.dataset_path = None
    config.checkpoint.enable = True                 # save checkpoints so we can resume later
    config.checkpoint.initial_load_path = None      # fresh run: do NOT load any checkpoint
    config.checkpoint.initial_load_in_hf = False
    config.checkpoint.initial_load_in_hf_quantized = False
    config.checkpoint.interval = 500               # Save at step interval
    config.checkpoint.keep_latest_k = 2            # keep only the 2 latest (each ~234G)
    # Distinct from gpt_oss_20b_lpt's dump_folder so the adahop and nolora runs
    # never share/overwrite each other's checkpoints.
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-adahop-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_adahop.yaml",),
    ],)
    return config

def gpt_oss_20b_adahop_hadamard() -> Trainer.Config:
    """Phase-2 Run A: AdaHOP with every slot forced to `hadamard` (no outlier
    extraction, no full_precision). Identical to gpt_oss_20b_adahop except the
    recipe's layer_transform_config maps all pattern-pairs to "hadamard".
    Isolates whether the AdaHOP training regression comes from mode SELECTION
    (S3) rather than the transform math. Distinct dump_folder so it never
    collides with the calibrated adahop or the nolora runs."""
    config = gpt_oss_20b_adahop()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-adahop-allhadamard-randomized-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_adahop_all_hadamard.yaml",),
    ],)
    return config
    
def gpt_oss_20b_lpt() -> Trainer.Config:
    config = gpt_oss_20b_pretrain_c4_megatron()
    config.dump_folder = "gpt_oss_20b-mi300-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-rank32-lr4e-4-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config

def gpt_oss_20b_lpt_madam() -> Trainer.Config:
    """Debug-model smoke test for the custom ``m_adam`` optimizer.

    ``lr`` maps to m_adam's additive (AdamW) branch ``lr_m`` and is driven
    by the usual LR scheduler; ``lr_e`` is the multiplicative (exponent)
    branch learning rate.
    """
    config = gpt_oss_20b_lpt()
    config.optimizer = MAdamOptimizersContainer.Config(
        lr=1e-4,
        lr_e=1e-4,
        tie_e_to_m=True,
        beta1=0.9,
        beta2=0.95,
        eps=1e-5,
        weight_decay_m=0.1,
    )
    return config


def gpt_oss_20b_lpt_madam_stable() -> Trainer.Config:
    """``gpt_oss_lpt_madam`` with a damped exponent branch.

    Same baseline-matched additive branch as ``gpt_oss_lpt_madam``, but the
    multiplicative (exponent) branch is turned down to suppress the loss
    spikes seen when it runs as loud as the mantissa branch:

    * ``lr_e`` dropped to 0.3x ``lr_m`` so the additive branch leads and the
      exponent only fine-tunes magnitudes, and
    * ``de_step_cap`` tightened from 0.5 to 0.2 (max per-step magnitude change
      ~2**0.2 ~= 1.15x instead of ~1.41x).
    """
    config = gpt_oss_20b_lpt_madam()
    config.optimizer.lr_e = 3e-5
    config.optimizer.de_step_cap = 0.2
    return config


def gpt_oss_20b_lpt_deosc() -> Trainer.Config:
    """weight deoscillation config."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-deosc"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_deosc.yaml",),
    ],)
    return config

def gpt_oss_20b_lpt_midmax() -> Trainer.Config:
    """gpt_oss_20b_lpt_c4 with midmax scale selection for MXFP4 quantization."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-midmax-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_midmax.yaml",),
    ],)
    return config

def gpt_oss_20b_lpt_uos() -> Trainer.Config:
    """gpt_oss_20b_lpt_c4 with uos scale selection for MXFP4 quantization."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-uos"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_uos.yaml",),
    ],)
    return config

def gpt_oss_20b_lpt_uos6() -> Trainer.Config:
    """gpt_oss_20b_lpt_c4 with uos scale selection for MXFP4 quantization."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-uos6"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_uos6.yaml",),
    ],)
    return config

def gpt_oss_20b_mxfp4_base() -> Trainer.Config:
    """baseline MXFP4 quantization."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-mxfp4-base"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/mxfp4_base.yaml",),
    ],)
    return config


def gpt_oss_20b_lpt_lowrank() -> Trainer.Config:
    """gpt_oss_20b_lpt_c4 with low-rank (lora_rank=32) correction for MXFP4 quantization."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-lr4e-4-lowrank-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_lowrank.yaml",),
    ],)
    return config
    
def gpt_oss_20b_lpt_1dw() -> Trainer.Config:
    """Plain MXFP4 with 1D-block weight quantization (use_2dblock_w: false):
    weights are scaled per-axis (1D macro blocks) instead of the baseline's 2D
    blocks. Identical to gpt_oss_20b_lpt otherwise. Own dump folder so it never
    collides with the 2D-weight baseline's checkpoints."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-1dw-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_1dw.yaml",),
    ],)
    return config

def gpt_oss_20b_lpt_fresh() -> Trainer.Config:
    """Plain MXFP4 baseline, fresh from step 1, own dump folder — the control for
    the weight-identity experiment (2D weight scaling ON). Distinct dump_folder
    from gpt_oss_20b_lpt so it never resumes an existing checkpoint."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-2dw-fresh-outputs"
    return config

def gpt_oss_20b_lpt_no2dw() -> Trainer.Config:
    """Plain MXFP4 with 2D weight scaling DISABLED (use_2dblock_w: false). Tests
    the AdaHOP root-cause hypothesis: breaking the baseline's single-2D-Q(W)
    identity (W re-quantized per-axis instead) should degrade plain MXFP4 toward
    AdaHOP's ~5.9 val@768. Fresh from step 1, own dump folder."""
    config = gpt_oss_20b_lpt()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-no2dw-fresh-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe_no2dw.yaml",),
    ],)
    return config

def gpt_oss_20b_grad_clip_lpt() -> Trainer.Config:
    """20b pretrain + MXFP4 + gradient clipping at the quantizer boundary."""
    config = gpt_oss_20b_pretrain()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-mxfp4-grad-clip-lr4e-4-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/grad_clip_lpt_recipe.yaml",),
    ],)
    return config


def gpt_oss_debugmodel_grad_clip_lpt() -> Trainer.Config:
    """Debugmodel + MXFP4 + gradient clipping. Use for config validation."""
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/grad_clip_lpt_recipe.yaml",),
    ],)
    return config


def gpt_oss_debugmodel_grad_clip_lpt_no_fsdp() -> Trainer.Config:
    """Debugmodel + MXFP4 + gradient clipping, FSDP disabled. Single GPU,
    no weight sharding — used to isolate whether FSDP is preventing the clip
    from firing."""
    config = gpt_oss_debugmodel_grad_clip_lpt()
    config.parallelism.data_parallel_shard_degree = 1
    return config


def gpt_oss_debugmodel_grad_clip_obs_lpt() -> Trainer.Config:
    """Debugmodel + MXFP4 + gradient clipping + DebugObserver. Produces tensor
    dumps under outputs/debug_obs_grad_clip_lpt.pt for comparison against the
    unclipped baseline."""
    config = gpt_oss_debugmodel()
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/grad_clip_obs_lpt_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_debugmodel_moe_pattern_obs() -> Trainer.Config:
    """Debugmodel + MXFP4 + per-expert MoE matmul outlier-pattern observer.
    Accumulates patterns over several steps (max_captures in the recipe) and
    dumps ./outputs/moe_patterns_rank*.pt with a per-expert majority vote.
    Use with COMM_MODE=local_tensor for a single-GPU smoke; compile stays off."""
    config = gpt_oss_debugmodel()
    config.training.steps = 10
    config.compile.enable = False
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/moe_pattern_observer_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_debugmodel_moe_pattern_obs_no_fsdp() -> Trainer.Config:
    """Single-GPU (no FSDP/EP/TP) debugmodel + MXFP4 + per-expert MoE matmul
    outlier-pattern observer. Loads a checkpoint produced by
    gpt_oss_debugmodel_lpt_1gpu_ckpt, runs ONE training step, and dumps
    ./outputs/moe_patterns_rank*.pt. Pairs with debug_stats_gpt_oss_1gpu.sh,
    which sets --checkpoint.initial_load_path to a step-N checkpoint."""
    config = gpt_oss_debugmodel_moe_pattern_obs()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.expert_parallel_degree = 1
    config.parallelism.expert_tensor_parallel_degree = 1
    config.parallelism.tensor_parallel_degree = 1
    # Load pretrained weights; the stats script clears any resumable state so the
    # step counter starts at 0 and the single observed step actually runs.
    config.checkpoint.enable = True
    return config


def gpt_oss_20b_moe_pattern_obs() -> Trainer.Config:
    """gpt_oss_20b: load a checkpoint, run ONE training step, and dump the
    per-expert MoE matmul outlier patterns. Intended flow: pretrain without the
    observer, point checkpoint.initial_load_path at a checkpoint, run this."""
    config = gpt_oss_20b_lpt()
    config.training.steps = 10
    config.compile.enable = False
    config.validator.enable = False
    config.checkpoint.enable = True                # load the pretrained checkpoint
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(
            recipe="./alto/models/gpt_oss/configs/moe_pattern_observer_recipe.yaml",
        ),
    ],)
    return config


def gpt_oss_20b_pretrain_c4() -> Trainer.Config:
    """gpt_oss_20b_pretrain using HuggingFace C4 dataset (bf16 baseline, no Megatron files required)."""
    config = gpt_oss_20b_pretrain()
    config.dump_folder = "gpt_oss_20b-pretrain-subset-bf16-c4-outputs"
    config.training.global_batch_size = 64
    config.optimizer.lr = 4e-4
    config.lr_scheduler.min_lr_factor = 0.04
    config.dataloader.dataset = "c4"
    config.dataloader.dataset_path = None
    config.validator.dataloader.dataset = "c4_validation"
    config.validator.dataloader.dataset_path = None
    config.checkpoint.initial_load_in_hf = True
    config.checkpoint.initial_load_in_hf_quantized = True
    return config

def gpt_oss_20b_lpt_c4() -> Trainer.Config:
    config = gpt_oss_20b_pretrain_c4()
    config.dump_folder = "gpt_oss_20b-mi300-pretrain-subset-mxfp4gemm_1d2d-hadamard-sr-rank32-c4-outputs"
    config.model_converters = ModelConvertersContainer.Config(converters=[
        ModelOptConverter.Config(recipe="./alto/models/gpt_oss/configs/lpt_recipe.yaml",),
    ],)
    return config