#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# GPTQ W4A8 quantization on Llama-3.1-8B
# Uses Hessian-based optimal weight quantization (4-bit per-group)
# with dynamic 8-bit activation quantization
#
# Usage:
#   bash examples/run_gptq.sh                          # 8 GPU, Llama-3.1-8B
#   NGPU=1 COMM_MODE=local_tensor bash examples/run_gptq.sh  # single-GPU debug
#   CONFIG=llama3_1b_gptq bash examples/run_gptq.sh   # switch to 1B model
rm -rf outputs/
set -ex

export CUDA_VISIBLE_DEVICES=0
export NGPU=${NGPU:-"1"}
export MODULE=${MODULE:-"llama3"}
export CONFIG=${CONFIG:-"llama3_8b_gptq"}

exec bash examples/run.sh "$@"
