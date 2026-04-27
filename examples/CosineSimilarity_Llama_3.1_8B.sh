#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Cosine similarity layer pruning on Llama-3.1-8B
#
# Usage:
#   bash examples/CosineSimilarity_Llama_3.1_8B.sh
#   NGPU=1 COMM_MODE=local_tensor bash examples/CosineSimilarity_Llama_3.1_8B.sh
rm -rf outputs/
set -ex

NGPU=${NGPU:-"1"}
export CUDA_VISIBLE_DEVICES=0
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_8b_cosine_similarity"}
COMM_MODE=${COMM_MODE:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@"
fi
