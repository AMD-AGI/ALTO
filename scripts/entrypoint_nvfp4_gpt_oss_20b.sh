#!/bin/bash
#
# K8s entrypoint for NVFP4 GPT-OSS-20B Low-Precision-Training.
# Mirrors scripts/entrypoint.sh (MXFP4) — same lifecycle, NVFP4-specific recipe.
#
# Overridable env vars (with defaults):
#   CONFIG        gpt_oss_20b_nvfp4_r8_lpt        trainer-config function in config_registry.py
#   GBS           16                              global batch size
#   OUTPUT_DIR    .../ALTO/nvfp4-outputs          dump folder root
#   RECIPE_FILE   /tmp/recipe.yaml                path to LPT yaml (auto-generated below
#                                                 if not provided; mount an external yaml
#                                                 here to override the recipe)
#   PET_NNODES    1                               # cluster nodes (K8s job sets this)
#   PET_NODE_RANK 0                               # current node rank
#   GPUS_PER_NODE 8                               # GPUs per node
#   MASTER_ADDR   localhost                       # rendezvous host
#   MASTER_PORT   1234
#
# Any config_registry.py / training-config field can be ADDITIONALLY overridden by
# appending --dotted.field value pairs after the script name (all forwarded via $@):
#
#   bash scripts/entrypoint_nvfp4_gpt_oss_20b.sh \
#        --training.steps 100 \
#        --debug.seed 42 \
#        --validator.freq 10 --validator.steps 10 \
#        --training.global-batch-size 32 \
#        --checkpoint.no-enable
#
# Switching to a different (experimental) NVFP4 recipe (mounted into the pod):
#
#   RECIPE_FILE=./alto/models/gpt_oss/configs/experimental/lpt_nvfp4_20b_first1_last2_bf16_recipe.yaml \
#   bash scripts/entrypoint_nvfp4_gpt_oss_20b.sh

set -ex

# 1) container-side filesystem setup (weka shared storage symlinks)
mkdir -p /workspace
ln -sfn /wekafs/hanwang2/huggingface /huggingface
ln -sfn /wekafs/hanwang2/workspace /workspace/workspace

cd /workspace/workspace/ALTO

# 2) editable installs (pip is idempotent; cheap on warm pod restarts)
pip install --no-build-isolation -e 3rdparty/torchtitan
pip install -e .

# 3) job parameters (env-var overridable)
RECIPE_FILE=${RECIPE_FILE:-"/tmp/recipe.yaml"}
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"gpt_oss"}
CONFIG=${CONFIG:-"gpt_oss_20b_nvfp4_r8_lpt"}     # canonical NVFP4 entry point
GBS=${GBS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-"/wekafs/zhitwang/alto_runs/gpt_oss_20b_nvfp4_outputs"}

# 4) inline NVFP4 LPT recipe (mirrors alto/models/gpt_oss/configs/
#    lpt_nvfp4_20b_r8_recipe.yaml after the PTS spec-fix landed on main).
#    Generated only when RECIPE_FILE is the default /tmp path AND does not
#    already exist, so external recipes mounted by the K8s job take priority.
if [ "$RECIPE_FILE" = "/tmp/recipe.yaml" ] && [ ! -f "$RECIPE_FILE" ]; then
  cat > "$RECIPE_FILE" <<'EOL'
training_stage:
  lpt_modifiers:
    LowPrecisionTrainingModifier:
      scheme: "nvfp4"
      targets: ["Linear", "GptOssGroupedExperts"]
      ignore: ["output", "re:.*\\.router\\.gate", "re:^layers\\.(21|22|23)(\\..*)?$"]
      use_2dblock_x: false
      use_2dblock_w: true
      use_hadamard: true
      use_sr_grad: true
      use_dge: false
      clip_mode: none
      two_level_scaling: tensorwise   # NVFP4 FP32 per-tensor scale (post-spec-fix)
      lora_rank: 0
EOL
fi

# 5) cluster env (PET_* is the elastic-launch convention K8s injects)
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export PET_NNODES=${PET_NNODES:-1}
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LOG_RANK=${LOG_RANK:-0}
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1      # required by NVFP4 Triton kernels

# 6) launch
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
  -m $TRAIN_FILE --module ${MODULE} --config ${CONFIG} \
  --dump-folder ${OUTPUT_DIR} \
  --model-converters.converters.0.recipe ${RECIPE_FILE} \
  --training.global-batch-size ${GBS} \
  "$@"
