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
# MODULE is either a model name (e.g. "llama3" -> alto.models.llama3.config_registry)
# or a fully qualified Python module containing the config function.
# TRAIN_FILE selects the trainer entrypoint (default: alto.train).
#
# Extra arguments are passed to the entrypoint; besides the TorchTitan loader
# options, alto.train accepts dotted config field overrides, e.g.
#   ./examples/run.sh --training.steps 10 --checkpointer.folder ckpt
#
# COMM_MODE="fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    Example: NGPU=32 COMM_MODE="fake_backend" ./examples/run.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
TRAIN_FILE=${TRAIN_FILE:-"alto.train"}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}

if [[ "$MODULE" != *.* ]]; then
    MODULE="alto.models.${MODULE}.config_registry"
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    if [ "$COMM_MODE" != "fake_backend" ]; then
        echo "COMM_MODE must be empty or fake_backend, got: ${COMM_MODE}" >&2
        exit 1
    fi
    echo "Running with fake process groups"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} --comm-backend fake "$@" --training.steps 1
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    HSA_NO_SCRATCH_RECLAIM=1 \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@"
fi
