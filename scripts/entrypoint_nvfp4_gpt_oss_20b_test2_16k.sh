#!/bin/bash
#
# K8s entrypoint: NVFP4-test-2 recipe, 16,000-step GPT-OSS-20B training.
#
# Goal: match mlperf-style long-horizon learning-rate envelope while only
# executing 16k actual training steps.  We accomplish this by setting:
#   * training.steps           = 16,000        (actual trainer loop length)
#   * lr_scheduler.total_steps = 1,200,000     (lr-curve denominator;
#                                               cosine traverses ~1.3% of
#                                               its arc, lr stays ~peak)
#   * lr_scheduler.warmup_steps = 128
#   * lr_scheduler.decay_ratio  = 1 - 128/1.2e6 (~0.99989)
#   * lr_scheduler.decay_type   = cosine
#   * lr_scheduler.min_lr_factor = 0.1
#
# All outputs (checkpoints / logs / tensorboard) land under
# /wekafs/zhitwang/alto_runs/...  Inputs (HF model, megatron c4 idx) are
# read-only references under /wekafs/hanwang2 and are NEVER written to.
#
# Overridable env vars (with defaults):
#   CONFIG          gpt_oss_20b_lpt              trainer-config entry
#                                                (inherits gpt_oss_20b_pretrain
#                                                schedule; we override knobs via
#                                                CLI below)
#   GBS             16
#   TRAINING_STEPS  16000                        loop cap
#   LR_PLAN_STEPS   1200000                      lr-scheduler denominator
#   CKPT_INTERVAL   1000                         ckpt every N steps
#   CKPT_KEEP       2                            keep_latest_k
#   VAL_FREQ        500
#   VAL_STEPS       64
#   RECIPE_FILE     /tmp/recipe.yaml             auto-generated below
#   RUN_TAG         nvfp4_test2_16k_<UTC stamp>
#   OUTPUT_DIR      /wekafs/zhitwang/alto_runs/$RUN_TAG
#   PET_NNODES / PET_NODE_RANK / GPUS_PER_NODE / MASTER_ADDR / MASTER_PORT
#                                                from K8s elastic-launch
#
# Any further training-config field can be appended after the script name
# as --dotted.field value pairs (forwarded via "$@").

set -ex

mkdir -p /workspace
ln -sfn /wekafs/hanwang2/huggingface /huggingface
ln -sfn /wekafs/hanwang2/workspace /workspace/workspace

# Prefer the dev workspace at /root/alto when present (warm pod / Cursor
# session); fall back to hanwang2's K8s read-only mount when not.
if [ -d /root/alto ]; then
  ALTO_REPO=${ALTO_REPO:-/root/alto}
else
  ALTO_REPO=${ALTO_REPO:-/workspace/workspace/ALTO}
fi
cd "$ALTO_REPO"

# Idempotent install: only pip-install if alto isn't already importable
# (avoids overwriting the existing dev install when we're on a warm pod).
if ! python3 -c "import alto" 2>/dev/null; then
  pip install --no-build-isolation -e 3rdparty/torchtitan
  pip install -e .
fi

RECIPE_FILE=${RECIPE_FILE:-"/tmp/recipe.yaml"}

# BF16 baseline mode: set RECIPE_FILE=NONE to skip the low-precision
# model-converter entirely (used by the sweep's bf16 reference run, which
# runs --config gpt_oss_20b_pretrain with no LpT modifier).
if [ "$RECIPE_FILE" = "/tmp/recipe.yaml" ] && [ ! -f "$RECIPE_FILE" ] && [ "$RECIPE_FILE" != "NONE" ]; then
  cat > "$RECIPE_FILE" <<'EOL'
training_stage:
  lpt_modifiers:
    LowPrecisionTrainingModifier:
      scheme: "nvfp4"
      targets: ["Linear", "GptOssGroupedExperts"]
      ignore:
        - "output"
        - "re:.*\\.router\\.gate"
        - "re:^layers\\.0(\\..*)?$"
        - "re:^layers\\.(22|23)(\\..*)?$"
      use_2dblock_x: false
      use_2dblock_w: true
      use_hadamard: true
      use_sr_grad: true
      use_dge: false
      clip_mode: none
      two_level_scaling: tensorwise
      lora_rank: 0
EOL
fi

TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"gpt_oss"}
CONFIG=${CONFIG:-"gpt_oss_20b_lpt"}
GBS=${GBS:-16}
# local_batch_size (micro-batch). grad_accum = GBS / (LOCAL_BS * dp_world).
# Default 1 preserves the original GBS=16 behavior; set LOCAL_BS=2 for the
# GBS=64 experiment to halve the grad-accumulation loop length.
LOCAL_BS=${LOCAL_BS:-1}

TRAINING_STEPS=${TRAINING_STEPS:-16000}
LR_PLAN_STEPS=${LR_PLAN_STEPS:-1200000}
LR_DECAY_RATIO=$(python3 -c "print(1 - 128 / ${LR_PLAN_STEPS})")

CKPT_INTERVAL=${CKPT_INTERVAL:-1000}
CKPT_KEEP=${CKPT_KEEP:-2}
VAL_FREQ=${VAL_FREQ:-500}
VAL_STEPS=${VAL_STEPS:-64}

RUN_TAG=${RUN_TAG:-"nvfp4_test2_16k_$(date -u +%Y%m%d_%H%M%S)"}
OUTPUT_DIR=${OUTPUT_DIR:-"/wekafs/zhitwang/alto_runs/${RUN_TAG}"}

# Converter arg is omitted entirely in BF16 baseline mode (RECIPE_FILE=NONE).
if [ "$RECIPE_FILE" = "NONE" ]; then
  CONVERTER_ARG=""
else
  CONVERTER_ARG="--model-converters.converters.0.recipe ${RECIPE_FILE}"
fi

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export PET_NNODES=${PET_NNODES:-1}
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
# Capture rank 0 (canonical metrics) AND rank 2 (historical SIGABRT
# location: previous May-20 1k longhorizon + May-24 16k both failed on
# local_rank 2; without rank-2 stderr we cannot tell whether it is a
# HIP runtime abort / NCCL async error / Triton assertion).
export LOG_RANK=${LOG_RANK:-0,2}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1
# Diagnostic env vars (post-mortem ready): if rank 2 SIGABRTs again we
# want a C++ stacktrace, NCCL trace dump, and Python faulthandler dump
# in the captured log.  These have negligible runtime cost.
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}
export TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}
export TORCH_NCCL_DESYNC_DEBUG=${TORCH_NCCL_DESYNC_DEBUG:-1}

HSA_NO_SCRATCH_RECLAIM=1 \
PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF \
torchrun \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${PET_NNODES}" \
  --node_rank "${PET_NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --local-ranks-filter ${LOG_RANK} \
  --role rank --tee 3 \
  -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} \
  --dump-folder ${OUTPUT_DIR} \
  ${CONVERTER_ARG} \
  --training.global-batch-size ${GBS} \
  --training.local-batch-size ${LOCAL_BS} \
  --training.steps ${TRAINING_STEPS} \
  --lr-scheduler.total-steps ${LR_PLAN_STEPS} \
  --lr-scheduler.warmup-steps 128 \
  --lr-scheduler.decay-ratio ${LR_DECAY_RATIO} \
  --lr-scheduler.decay-type cosine \
  --lr-scheduler.min-lr-factor 0.1 \
  --checkpoint.interval ${CKPT_INTERVAL} \
  --checkpoint.keep-latest-k ${CKPT_KEEP} \
  --validator.freq ${VAL_FREQ} \
  --validator.steps ${VAL_STEPS} \
  --metrics.log-freq 1 \
  "$@"
