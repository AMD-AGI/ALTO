#!/bin/bash

set -ex

mkdir /workspace
ln -s /wekafs/hanwang2/huggingface /huggingface
ln -s /wekafs/hanwang2/workspace /workspace/workspace

cd /workspace/workspace/ALTO

pip install --no-build-isolation -e 3rdparty/torchtitan
pip install -e .

RECIPE_FILE="/tmp/recipe.yaml"
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"gpt_oss"}
CONFIG=${CONFIG:-"gpt_oss_20b_lpt"}
GBS=${GBS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/workspace/ALTO/debug-outputs"}

cat > $RECIPE_FILE <<EOL
training_stage:
  lpt_modifiers:
    LowPrecisionTrainingModifier:
      scheme: "mxfp4"
      targets: ["Linear", "GptOssGroupedExperts"]
      ignore: ["output", "re:.*\\\\.router\\\\.gate"]
      use_2dblock_x: false
      use_2dblock_w: true
      use_hadamard: true
      use_sr_grad: true
      use_dge: false
      clip_mode: none
      two_level_scaling: none
      lora_rank: 0
EOL

# Set cluster ENV
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export PET_NNODES=${PET_NNODES:-1}
export PET_NODE_RANK=${PET_NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export LOG_RANK=${LOG_RANK:-0}
export PYTORCH_ALLOC_CONF="expandable_segments:True"

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
