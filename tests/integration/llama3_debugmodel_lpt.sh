#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=llama3_debugmodel_lpt \
HSA_NO_SCRATCH_RECLAIM=1 \
./examples/run.sh
