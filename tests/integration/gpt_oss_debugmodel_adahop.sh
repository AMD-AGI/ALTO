#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Smoke test: gpt_oss debug model under MXFP4 + AdaHOP + lora_rank=32.
# Runs 30 calibration steps (Phase A) followed by 5 post-calibration training
# steps (Phase B). Confirms that DecomposedLinear weights are discovered by
# _collect_calibration_wrappers and that the Phase-B re-swap fires correctly.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=4 \
MODULE=gpt_oss \
CONFIG=gpt_oss_debugmodel_adahop \
./examples/run.sh
