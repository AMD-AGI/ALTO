#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$workspace:$PYTHONPATH
workspace=/wekafs/guanchen/Model-Optimizer

task_name=llama-cosine-similarity
config=${workspace}/configs/pruning/llama-cosine-similarity.yml


nnodes=1
nproc_per_node=1


torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
${workspace}/src/__main__.py --config $config | tee ${task_name}.log