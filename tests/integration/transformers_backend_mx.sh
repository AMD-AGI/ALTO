#!/bin/bash

set -e

: "${HF_MODEL_PATH:?Set HF_MODEL_PATH to an absolute local HF model directory}"
export HF_MODEL_PATH

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/../.."

for backend_config in hf_backend_bf16 hf_backend_mx6_wa hf_backend_mx9_wa; do
  NGPU=1 \
  MODULE=transformers_modeling_backend \
  CONFIG="$backend_config" \
  ./examples/run.sh "$@"
done
