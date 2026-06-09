#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# MX9 W8A8 dynamic fake-quant validation on Llama-3.2-1B
# Weight and input activations are quantized dynamically through
# alto/models/llama3/configs/mx9_wa_recipe.yaml
#
# Usage:
#   bash examples/llama3.2_1b_mx9.sh
#   VALIDATOR_STEPS=100 bash examples/llama3.2_1b_mx9.sh
#   CONFIG=llama3_1b bash examples/llama3.2_1b_mx9.sh  # BF16 baseline
rm -rf outputs/
set -ex

NGPU=${NGPU:-"1"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_1b_mx9_wa"}
COMM_MODE=${COMM_MODE:-""}

MODEL_PATH=${MODEL_PATH:-"/wekafs/jiarwang/Llama-3.2-1B"}
VALIDATOR_STEPS=${VALIDATOR_STEPS:-"10"}
CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-"/wekafs/jiarwang/mx9_e2e_logs/ckpt_${CONFIG}_$(date +%Y%m%d_%H%M%S)"}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m ${TRAIN_FILE} \
        --module ${MODULE} \
        --config ${CONFIG} \
        --hf_assets_path "${MODEL_PATH}" \
        --checkpoint.initial_load_path "${MODEL_PATH}" \
        --checkpoint.folder "${CHECKPOINT_FOLDER}" \
        --validator.steps "${VALIDATOR_STEPS}" \
        "$@" \
        --comm.mode=${COMM_MODE} \
        --training.steps 1
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
        -m ${TRAIN_FILE} \
        --module ${MODULE} \
        --config ${CONFIG} \
        --hf_assets_path "${MODEL_PATH}" \
        --checkpoint.initial_load_path "${MODEL_PATH}" \
        --checkpoint.folder "${CHECKPOINT_FOLDER}" \
        --validator.steps "${VALIDATOR_STEPS}" \
        "$@"
fi
