#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=llama3_debugmodel_lpt \
./examples/run.sh
