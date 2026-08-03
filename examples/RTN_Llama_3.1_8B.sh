#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

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
rm -rf outputs/
set -ex

export CUDA_VISIBLE_DEVICES=0
export NGPU=${NGPU:-"1"}
export MODULE=${MODULE:-"llama3"}
export CONFIG=${CONFIG:-"llama3_8b_rtn"}

exec bash examples/run.sh "$@"
