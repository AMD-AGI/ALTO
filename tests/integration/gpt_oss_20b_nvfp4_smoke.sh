#!/bin/bash
# One-step GPT-OSS-20B NVFP4 smoke using WekaFS model and C4 paths.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \
NGPU=8 \
MODULE=gpt_oss \
CONFIG=gpt_oss_20b_nvfp4_smoke \
./examples/run.sh
