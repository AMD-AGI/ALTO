#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$workspace:$PYTHONPATH
workspace=/group/ossdphi_algo_scratch_13/guanchen/AMD-Model-Optimizer

task_name=llama-wanda-demo
config=${workspace}/configs/sparsification/llama-wanda-demo.yml


nnodes=1
nproc_per_node=1


torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
${workspace}/src/__main__.py --config $config | tee ${task_name}.log