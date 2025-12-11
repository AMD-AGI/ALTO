#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

workspace=/group/ossdphi_algo_scratch_13/guanchen/AMD-Model-Optimizer
export PYTHONPATH=$workspace:$PYTHONPATH


task_name=phi_awq_w_a
config=${workspace}/configs/sparsification/llama-wanda.yml


nnodes=1
nproc_per_node=1


torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
${workspace}/src/__main__.py --config $config | tee log.txt