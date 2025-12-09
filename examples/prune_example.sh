#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

workspace=/group/ossdphi_algo_scratch_13/guanchen/AMD-Model-Optimizer
export PYTHONPATH=$workspace:$PYTHONPATH

task_name=phi_awq_w_a
config=${workspace}/configs/sparsification/llama-sparsegpt.yml

nnodes=1
nproc_per_node=1


find_unused_port() {
    while true; do
        port=$(shuf -i 10000-60000 -n 1)
        if ! ss -tuln | grep -q ":$port "; then
            echo "$port"
            return 0
        fi
    done
}
UNUSED_PORT=$(find_unused_port)


MASTER_ADDR=127.0.0.1
MASTER_PORT=$UNUSED_PORT
task_id=$UNUSED_PORT


torchrun \
--nnodes $nnodes \
--nproc_per_node $nproc_per_node \
--rdzv_id $task_id \
--rdzv_backend c10d \
--rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
${workspace}/src/__main__.py --config $config --task_id $task_id | tee log.txt