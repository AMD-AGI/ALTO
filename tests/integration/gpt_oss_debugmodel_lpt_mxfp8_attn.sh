#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=${NGPU:-1} \
MODULE=gpt_oss \
CONFIG=gpt_oss_debugmodel_lpt_mxfp8_attn \
./examples/run.sh
