#!/usr/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# MX6 / MX9 dynamic validation on Llama-3.2-1B
#
#   MX=6 -> W5A5, alto/models/llama3/configs/mx6_wa_recipe.yaml
#   MX=9 -> W8A8, alto/models/llama3/configs/mx9_wa_recipe.yaml
#
# Weight and input activations are quantized dynamically. format=="mx6"/"mx9"
# dispatches to the packed Triton kernel (convert_to_mx / convert_from_mx) in
# alto/models/patcher.py, not the fake-quantize emulation.
#
# Usage (MX and MODEL_PATH are both required):
#   MODEL_PATH=/path/to/Llama-3.2-1B MX=9 bash examples/llama3.2_1b_mx.sh
#   MODEL_PATH=/path/to/Llama-3.2-1B MX=6 bash examples/llama3.2_1b_mx.sh
#   MODEL_PATH=/path/to/Llama-3.2-1B MX=6 VALIDATOR_STEPS=100 bash examples/llama3.2_1b_mx.sh
#   MODEL_PATH=/path/to/Llama-3.2-1B MX=6 VALIDATOR_STEPS=-1 bash examples/llama3.2_1b_mx.sh  # full validation set once
#   MODEL_PATH=/path/to/Llama-3.2-1B CONFIG=llama3_1b bash examples/llama3.2_1b_mx.sh         # BF16 baseline
#
# Only the recipe defaults live here; examples/run.sh is the shared launcher and
# documents the remaining env vars (LOG_RANK, COMM_MODE, TRAIN_FILE, ...).
rm -rf outputs/
set -ex

MODEL_PATH=${MODEL_PATH:-""}
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: MODEL_PATH must be set to your local Llama-3.2-1B directory, e.g." >&2
    echo "       MODEL_PATH=/path/to/Llama-3.2-1B MX=9 bash $0" >&2
    exit 1
fi

# CONFIG overrides MX entirely (e.g. CONFIG=llama3_1b for the BF16 baseline);
# MX is only consulted when CONFIG is left at its default.
MX=${MX:-""}
if [ -z "${CONFIG:-}" ]; then
    case "${MX}" in
        6|9) CONFIG="llama3_1b_mx${MX}_wa" ;;
        *)
            echo "ERROR: MX must be set to 6 or 9, e.g." >&2
            echo "       MODEL_PATH=/path/to/Llama-3.2-1B MX=9 bash $0" >&2
            exit 1
            ;;
    esac
fi

VALIDATOR_STEPS=${VALIDATOR_STEPS:-"10"}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
export NGPU=${NGPU:-"1"}
export MODULE=${MODULE:-"llama3"}
export CONFIG

CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-"./outputs/ckpt_${CONFIG}_$(date +%Y%m%d_%H%M%S)"}

exec bash examples/run.sh \
    --hf_assets_path "${MODEL_PATH}" \
    --checkpoint.initial_load_path "${MODEL_PATH}" \
    --checkpoint.folder "${CHECKPOINT_FOLDER}" \
    --validator.steps "${VALIDATOR_STEPS}" \
    "$@"
