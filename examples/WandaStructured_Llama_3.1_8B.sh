#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Wanda structured MLP pruning on Llama-3.1-8B
#
# Usage:
#   bash examples/WandaStructured_Llama_3.1_8B.sh
#   NGPU=1 COMM_MODE=local_tensor bash examples/WandaStructured_Llama_3.1_8B.sh
rm -rf outputs/
set -ex

export CUDA_VISIBLE_DEVICES=0
export NGPU=${NGPU:-"1"}
export MODULE=${MODULE:-"llama3"}
export CONFIG=${CONFIG:-"llama3_8b_wanda_structured"}

exec bash examples/run.sh "$@"
