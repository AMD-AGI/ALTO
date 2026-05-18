#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=1 \
MODULE=llama3 \
CONFIG=llama3_1b_opt \
HSA_NO_SCRATCH_RECLAIM=1 \
./examples/run.sh \
  --hf_assets_path=/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 \
  --checkpoint.initial_load_path=/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08
