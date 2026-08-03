#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# AWQ W4A8 quantization on Llama-3.1-8B
# Uses activation-weighted scaling with grid search (4-bit per-group)
# with dynamic 8-bit activation quantization
#
# Usage:
#   bash examples/run_awq.sh                          # 8 GPU, Llama-3.1-8B
#   NGPU=1 COMM_MODE=local_tensor bash examples/run_awq.sh  # single-GPU debug
#   CONFIG=llama3_1b_awq bash examples/run_awq.sh    # switch to 1B model
rm -rf outputs/
set -ex

export CUDA_VISIBLE_DEVICES=0
export NGPU=${NGPU:-"1"}
export MODULE=${MODULE:-"llama3"}
export CONFIG=${CONFIG:-"llama3_8b_awq"}

exec bash examples/run.sh "$@"
