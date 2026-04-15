#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Modifications by Advanced Micro Devices, Inc. are licensed under the MIT License
# (see LICENSE in the root of this repository).
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Original portions are licensed under the BSD 3-Clause License (see upstream PyTorch/torchtitan licensing).
#
# SPDX-License-Identifier: BSD-3-Clause AND MIT

set -ex

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@"
fi
