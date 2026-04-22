#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

NGPU=2 \
MODULE=llama3 \
CONFIG=instella_3b_opt \
./examples/run.sh \
  --hf_assets_path=/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e \
  --checkpoint.initial_load_path=/huggingface/hub/models--amd--Instella-3B-Stage1/snapshots/cb33253ab0a5b9f2ea0b98f3edd818d46454580e
