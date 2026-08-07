#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Fast-iteration smoke for AdaHOP two-phase swap: 3 calibration steps
# followed by 5 post-Phase-B training steps. Use to debug the calibration
# → re-swap → frozen-mode pipeline without waiting 30 steps of calibration.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=llama3_debugmodel_adahop_short \
./examples/run.sh
