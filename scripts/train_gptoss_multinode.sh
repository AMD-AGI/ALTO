#!/usr/bin/env bash
#SBATCH -A amd-arad
#SBATCH -p amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --job-name=alto-gptoss20b-multinode
#SBATCH --output=alto-gptoss20b-multinode-%j.out
#SBATCH --requeue
#
# Multi-node ALTO GPT-OSS 20B training.
#
# Runs one Docker container per node (from the locally-built alto:multinode
# image) and launches a single torchrun job spanning all nodes, rendezvousing
# on the first allocated node.
#
#   sbatch scripts/train_gptoss_20b.sh
#
# Everything below is env-overridable, e.g.:
#   TRAINING_STEPS=50 CONFIG=gpt_oss_debugmodel sbatch scripts/train_gptoss_20b.sh
#
# NOTE: CHECKPOINT_DIR, LOG_DIR, HF_HOME_DIR and ALTO_DIR must live on a
# filesystem shared across all nodes (e.g. $HOME or /shared) so every rank sees
# the same repo, model assets and checkpoints.

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

### Machine-specific args
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}" # HF model / dataset cache
DATA_DIR="${DATA_DIR:-/shared_inference}"              # data dir exposed to container
HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf.env}"           # .env file with raw HF token

### Run-specific args
# *NOTE*: if you cloned multiple copies of this repo, make sure the path below is correct
ALTO_DIR="${ALTO_DIR:-$HOME/lpt_branch/ALTO}"         # repo dir exposed to container
CONFIG="${CONFIG:-gpt_oss_20b_pretrain_c4}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ALTO_DIR/gptoss_chkpt/${CONFIG}_$RUN_ID}"
LOG_DIR="${LOG_DIR:-$ALTO_DIR/logs}"

### Other modifiable args
MODULE="${MODULE:-gpt_oss}"
TRAINING_STEPS="${TRAINING_STEPS:-15000}"
MODEL_DIR="${MODEL_DIR:-$HF_HOME_DIR/models/gpt-oss-20b}"

### Docker image built from Dockerfile.multinode
IMAGE="${IMAGE:-alto:multinode}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.multinode}"

# -----------------------------------------------------------------------------
# Build the multinode image on every node
# -----------------------------------------------------------------------------
# Docker images are local to each node's daemon, so the image must be built on
# every allocated node -- building only on the batch node would leave the other
# nodes unable to find alto:multinode at `docker run` time.

cd "$ALTO_DIR"

echo "[build] Building $IMAGE from $DOCKERFILE on all nodes ..."
srun --ntasks-per-node=1 \
    bash -c "cd '$ALTO_DIR' && docker build -f '$DOCKERFILE' -t '$IMAGE' ."

# -----------------------------------------------------------------------------
# Rendezvous / bookkeeping
# -----------------------------------------------------------------------------

# First allocated node becomes the torchrun rendezvous host.
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"

mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$HF_HOME_DIR"

echo "=== ALTO GPT-OSS 20B (multinode) ==="
echo "Nodes:          $(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd, -)"
echo "Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "Image:          $IMAGE"
echo "Config:         $CONFIG"
echo "GPUs per node:  $GPUS_PER_NODE"
echo "Total GPUs:     $((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))"
echo "Training steps: $TRAINING_STEPS"
echo "Model dir:      $MODEL_DIR"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "Logs:           $LOG_DIR"
echo

# -----------------------------------------------------------------------------
# Fetch tokenizer / model config once (shared FS, so all ranks reuse it)
# -----------------------------------------------------------------------------
if [[ -f "$MODEL_DIR/tokenizer.json" ]]; then
    echo "[model] Tokenizer already present at $MODEL_DIR, skipping download."
else
    echo "[model] Downloading tokenizer / config to $MODEL_DIR ..."
    docker run --rm \
        --user "$(id -u):$(id -g)" \
        --network host \
        --env-file "$HF_ENV_FILE" \
        -v "$HOME:$HOME" \
        -v "$HF_HOME_DIR:/hf_home" \
        -v /etc/passwd:/etc/passwd:ro \
        -v /etc/group:/etc/group:ro \
        -e HOME="$HOME" \
        -e USER="$(id -un)" \
        -e HF_HOME=/hf_home \
        "$IMAGE" \
        hf download openai/gpt-oss-20b \
            --include "tokenizer*" "special_tokens_map.json" "config.json" \
            --local-dir "$MODEL_DIR"
fi

# Export everything the per-node srun step needs.
export IMAGE ALTO_DIR HF_HOME_DIR DATA_DIR HF_ENV_FILE
export CONFIG MODULE TRAINING_STEPS MODEL_DIR CHECKPOINT_DIR LOG_DIR RUN_ID
export GPUS_PER_NODE MASTER_ADDR MASTER_PORT

# -----------------------------------------------------------------------------
# Launch: one srun task -> one container -> torchrun per node
# -----------------------------------------------------------------------------
srun --kill-on-bad-exit=1 bash -c '
    set -euo pipefail

    CONTAINER="alto_${SLURM_JOB_ID}_${SLURM_NODEID}"
    NODE_LOG="$LOG_DIR/gpt_oss_20b-${RUN_ID}-node${SLURM_NODEID}.log"

    cleanup() {
        docker stop --time 10 "$CONTAINER" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM

    # Assemble docker args, adding hardware resources only when present on this node.
    docker_args=(
        --rm
        --name "$CONTAINER"
        --user "$(id -u):$(id -g)"
        --network host
        --ipc host
        --shm-size 128g
        --ulimit memlock=-1
        --cap-add SYS_PTRACE
        --security-opt seccomp=unconfined
        --env-file "$HF_ENV_FILE"
        -v "$HOME:$HOME"
        -v "$ALTO_DIR:/alto"
        -v "$DATA_DIR:$DATA_DIR"
        -v "$HF_HOME_DIR:/hf_home"
        -v "$CHECKPOINT_DIR:$CHECKPOINT_DIR"
        -v /shared:/shared
        -v /etc/passwd:/etc/passwd:ro
        -v /etc/group:/etc/group:ro
        -e HOME="$HOME"
        -e USER="$(id -un)"
        -e HF_HOME=/hf_home
        -e HF_DATASETS_CACHE=/hf_home/datasets
        -e TRITON_CACHE_DIR=/tmp/triton_cache
        -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
        -e PYTHONNOUSERSITE=1
        -e TRANSFORMERS_OFFLINE=1
        -e PYTORCH_ALLOC_CONF=expandable_segments:True
        -w /alto
    )

    for device in /dev/kfd /dev/dri /dev/infiniband; do
        [[ -e "$device" ]] && docker_args+=(--device "$device")
    done

    for group in render video; do
        gid="$(getent group "$group" | cut -d: -f3 || true)"
        [[ -n "$gid" ]] && docker_args+=(--group-add "$gid")
    done

    echo "[node $SLURM_NODEID] launching torchrun on $(hostname) -> $NODE_LOG"

    docker run "${docker_args[@]}" "$IMAGE" \
        torchrun \
            --nnodes "$SLURM_JOB_NUM_NODES" \
            --nproc-per-node "$GPUS_PER_NODE" \
            --node-rank "$SLURM_NODEID" \
            --master-addr "$MASTER_ADDR" \
            --master-port "$MASTER_PORT" \
            --local-ranks-filter 0 \
            --tee 3 \
            -m alto.train \
            --module "$MODULE" \
            --config "$CONFIG" \
            --training.steps "$TRAINING_STEPS" \
            --comm.init_timeout_seconds 1800 \
            --hf_assets_path "$MODEL_DIR" \
            --dump_folder "$CHECKPOINT_DIR" \
            --profiling.enable_profiling \
            --profiling.profile_freq 1000 \
            --profiling.profiler_warmup 3 \
            --profiling.profiler_active 1 \
        2>&1 | tee "$NODE_LOG"
'

echo "[train] Multinode run complete."
