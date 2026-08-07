#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
#
# Phase-3 smoke: llama3 debug model under MXFP4 + AdaHOP two-phase swap.
# Runs 30 calibration steps (Phase A) followed by 5 post-calibration training
# steps (Phase B). Confirms the calibration callback + backward hook + re-swap
# pipeline runs end-to-end without crashing.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=llama3_debugmodel_adahop \
./examples/run.sh
