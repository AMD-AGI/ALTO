#!/usr/bin/env bash
# Run ALTO GPT-OSS 20B BF16 training on the current node.
#
# Example:
#   NGPU=4 CONFIG=gpt_oss_debugmodel TRAINING_STEPS=20 \
#       bash ~/ALTO/train_gptoss.sh

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ALTO_DIR="${ALTO_DIR:-$HOME/ALTO}"
IMAGE="${IMAGE:-wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch}"
CONTAINER="${CONTAINER:-alto_gpt_oss_bf16}"

NGPU="${NGPU:-8}"
MODULE="${MODULE:-gpt_oss}"
CONFIG="${CONFIG:-gpt_oss_20b_pretrain_c4}"
TRAINING_STEPS="${TRAINING_STEPS:-15000}"

MODEL_REPO="${MODEL_REPO:-openai/gpt-oss-20b}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/gpt-oss-20b}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}"
HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf.env}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/gptoss_chkpt/gpt_oss_20b-pretrain-bf16}"
LOG_FILE="${LOG_FILE:-$ALTO_DIR/gpt_oss_20b-bf16.log}"

# Comma-separated host directories mounted at the same path in the container.
# Example: EXTRA_MOUNTS=/shared,/shared_rccl
EXTRA_MOUNTS="${EXTRA_MOUNTS:-/shared_rccl}"

# Hardware resources are added only when they exist.
DEVICE_PATHS="${DEVICE_PATHS:-/dev/kfd /dev/dri /dev/infiniband}"
DEVICE_GROUPS="${DEVICE_GROUPS:-render video}"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

mkdir -p \
    "$MODEL_DIR" \
    "$HF_HOME_DIR" \
    "$CHECKPOINT_DIR" \
    "$(dirname "$LOG_FILE")"

echo "=== ALTO GPT-OSS 20B BF16 baseline ==="
echo "Node:           $(hostname)"
echo "Image:          $IMAGE"
echo "Config:         $CONFIG"
echo "GPUs:           $NGPU"
echo "Training steps: $TRAINING_STEPS"
echo "Model:          $MODEL_DIR"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "Log:            $LOG_FILE"
echo

docker pull "$IMAGE"

docker_args=(
    -d
    --rm
    --name "$CONTAINER"
    --user "$(id -u):$(id -g)"
    --network host
    --ipc host
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v "$HOME:$HOME"
    -v "$ALTO_DIR:/alto"
    -v "$MODEL_DIR:$MODEL_DIR"
    -v "$HF_HOME_DIR:/hf_home"
    -v "$CHECKPOINT_DIR:$CHECKPOINT_DIR"
    -v /etc/passwd:/etc/passwd:ro
    -v /etc/group:/etc/group:ro
    -e HOME="$HOME"
    -e USER="$(id -un)"
    -e HF_HOME=/hf_home
    -e HF_DATASETS_CACHE=/hf_home/datasets
    -e TRITON_CACHE_DIR=/tmp/triton_cache
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
)

if [[ -f "$HF_ENV_FILE" ]]; then
    docker_args+=(--env-file "$HF_ENV_FILE")
fi

for device in $DEVICE_PATHS; do
    if [[ -e "$device" ]]; then
        docker_args+=(--device "$device")
    fi
done

for group in $DEVICE_GROUPS; do
    gid="$(getent group "$group" | cut -d: -f3 || true)"

    if [[ -n "$gid" ]]; then
        docker_args+=(--group-add "$gid")
    fi
done

if [[ -n "$EXTRA_MOUNTS" ]]; then
    IFS=',' read -r -a mount_dirs <<< "$EXTRA_MOUNTS"

    for directory in "${mount_dirs[@]}"; do
        if [[ ! -d "$directory" ]]; then
            echo "Missing mount directory: $directory" >&2
            exit 1
        fi

        docker_args+=(-v "$directory:$directory")
    done
fi

docker run "${docker_args[@]}" "$IMAGE" sleep infinity

cleanup() {
    echo "[train] Stopping container $CONTAINER ..."
    docker stop "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Model and dependencies
# -----------------------------------------------------------------------------

echo "[model] Ensuring $MODEL_REPO is available at $MODEL_DIR ..."

docker exec "$CONTAINER" \
    hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"

echo "[train] Installing dependencies ..."

docker exec "$CONTAINER" \
    python3 -m pip install -q torchao

docker exec "$CONTAINER" \
    python3 -m pip install -q \
        --no-build-isolation \
        --no-deps \
        -e /alto/3rdparty/torchtitan

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

echo "[train] Launching $CONFIG for $TRAINING_STEPS steps on $NGPU GPUs ..."

docker exec \
    -w /alto \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True \
    -e TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    torchrun \
        --standalone \
        --nproc_per_node "$NGPU" \
        --local-ranks-filter 0 \
        --tee 3 \
        -m alto.train \
        --module "$MODULE" \
        --config "$CONFIG" \
        --training.steps "$TRAINING_STEPS" \
        --comm.init_timeout_seconds 1800 \
        --hf_assets_path "$MODEL_DIR" \
        --checkpoint.interval 1000 \
        --checkpoint.keep_latest_k 2 \
        --dump_folder "$CHECKPOINT_DIR" \
        --profiling.enable_profiling \
        --profiling.profile_freq 1000 \
        --profiling.profiler_warmup 3 \
        --profiling.profiler_active 1 \
    2>&1 | tee "$LOG_FILE"

echo
echo "[train] Run complete."