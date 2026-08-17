#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Step-1 baseline companion to llama3_debugmodel_lpt.sh: runs the same llama3
# debug config without the LowPrecisionTrainingModifier, so loss/throughput
# can be A/B'd against the mxfp4 path.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=llama3_debugmodel \
./examples/run.sh \
  --training.steps 10
