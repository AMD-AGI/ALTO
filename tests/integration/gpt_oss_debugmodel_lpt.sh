#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=gpt_oss \
CONFIG=gpt_oss_debugmodel_lpt \
./examples/run.sh
