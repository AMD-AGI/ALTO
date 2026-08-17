#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Phase-3 real-model run: Llama-3.2-1B under MXFP4 + AdaHOP two-phase swap.
# 1000 training steps (calibration window controlled by the recipe).
# Mirrors the llama3_1b_lpt smoke but with the AdaHOP recipe — same model,
# same data, same target subset (Linear modules, output excluded).

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=${NGPU:-8} \
MODULE=llama3 \
CONFIG=llama3_1b_adahop \
./examples/run.sh "$@"
