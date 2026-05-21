#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \
NGPU=2 \
MODULE=gpt_oss \
CONFIG=gpt_oss_debugmodel_nvfp4_lpt \
./examples/run.sh
