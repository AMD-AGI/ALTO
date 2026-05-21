#!/bin/bash
# 8-GPU EP sanity smoke for NVFP4 GPT-OSS debug model.
#
# Phase 2 step 8 of the NVFP4 + GPT-OSS QAT plan: ensure the
# NVFP4TrainingWeightWrapperTensor survives DTensor / EP all-to-all and
# reaches the grouped GEMM dispatch path on every expert.

SCRIPT_DIR=$(dirname "$0")
cd $SCRIPT_DIR/../..

TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \
NGPU=8 \
MODULE=gpt_oss \
CONFIG=gpt_oss_debugmodel_nvfp4_lpt \
./examples/run.sh \
    --training.steps 50 \
    --parallelism.expert_parallel_degree 8 \
    --parallelism.tensor_parallel_degree 1 \
    --parallelism.expert_tensor_parallel_degree 1
