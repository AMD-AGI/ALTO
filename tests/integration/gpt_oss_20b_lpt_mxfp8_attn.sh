#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

# Real GPT-OSS-20B uses expert_parallel_degree=8 in the base pretrain config.
NGPU=${NGPU:-8} \
MODULE=gpt_oss \
CONFIG=gpt_oss_20b_lpt_mxfp8_attn \
./examples/run.sh
