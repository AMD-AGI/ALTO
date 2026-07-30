#!/usr/bin/env bash
#SBATCH -A amd-arad
#SBATCH -p amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --job-name=alto-multinode
#SBATCH --output=alto-multinode-%j.out
#SBATCH --requeue

set -euo pipefail

IMAGE="alto:multinode"
ALTO_DIR="$HOME/lpt_branch/ALTO"
GPUS_PER_NODE=8

# First allocated node becomes the torchrun rendezvous host.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

export IMAGE ALTO_DIR GPUS_PER_NODE MASTER_ADDR MASTER_PORT

echo "Nodes: $(scontrol show hostnames "$SLURM_JOB_NODELIST")"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

# One srun task, and therefore one Docker container, per node.
srun --kill-on-bad-exit=1 bash -c '
    set -euo pipefail

    CONTAINER="alto_${SLURM_JOB_ID}_${SLURM_NODEID}"

    cleanup() {
        docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM

    docker run --rm \
        --name "$CONTAINER" \
        --network=host \
        --ipc=host \
        --shm-size=128g \
        --ulimit memlock=-1 \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        -v /dev/infiniband:/dev/infiniband \
        -v "$ALTO_DIR:/alto" \
        -v /shared:/shared \
        -w /alto \
        "$IMAGE" \
        torchrun \
            --nnodes="$SLURM_JOB_NUM_NODES" \
            --nproc-per-node="$GPUS_PER_NODE" \
            --node-rank="$SLURM_NODEID" \
            --master-addr="$MASTER_ADDR" \
            --master-port="$MASTER_PORT" \
            rdma_tests/test_rdma_allreduce.py
'