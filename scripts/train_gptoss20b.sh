#!/usr/bin/env bash
# Run ALTO GPT-OSS 20B BF16 training on the current node.
#
# Example:
#   NGPU=4 CONFIG=gpt_oss_debugmodel TRAINING_STEPS=20 \
#       bash ~/ALTO/train_gptoss.sh

######## STEP 0: Install ALTO repository and update ALTO_DIR below
# git clone --recurse-submodules https://github.com/AMD-AGI/ALTO.git


######## STEP 1: Download the C4 dataset
######## make sure to update config_registry.py with appropriate data location
###  OPTION 1:
# # Create desired download directory with the right permission 
# cd /data/gpt_oss_20b
# # Download training and validation data
# bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
#     -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
### OPTION 2: 
# C4_CACHE="$HF_HOME_SHARED/datasets/allenai___c4"
# if [ -d "$C4_CACHE" ] && [ -n "$(ls -A "$C4_CACHE" 2>/dev/null)" ]; then
#     echo "[train] C4 dataset already cached, skipping download."
# else
#     echo "[train] Downloading C4 dataset (this may take a while) ..."
#     docker exec "$CONTAINER" bash -c "
#         python3 -c \"
# from datasets import load_dataset
# load_dataset('allenai/c4', 'en', split='train')
# load_dataset('allenai/c4', 'en', split='validation')
#         \"
#     "
# fi

####### Viewing Loss Curves
# tensorboard events are saved in the checkpointing directory, one can
# view these by using the following command:
# tensorboard --logdir $CHECKPOINT_DIR --host 127.0.0.1 --port 6006
#
# If running on remote machine, you will want to forward the port to the local machine:
# ssh -L 6006:localhost:6006 nfrumkin@useocpslog-002
 
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

### Machine-specific args
NGPU="${NGPU:-8}"
HF_HOME_DIR="${HF_HOME_DIR:-$HOME/.cache/huggingface}" # HF model location
DATA_DIR="${DATA_DIR:-/shared_inference}" # exposte data dir to container
HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf.env}" # .env file has raw HF access token

### Run-specific args
# *NOTE*: if you cloned multiple copies of this repo, make sure the path below is correct
ALTO_DIR="${ALTO_DIR:-$HOME/lpt_branch/ALTO}" # expose repo dir to container 
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ALTO_DIR/gptoss_chkpt/gpt_oss_20b-pretrain-bf16}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
LOG_FILE="${LOG_FILE:-$ALTO_DIR/logs/gpt_oss_20b-bf16-$RUN_ID.log}" # log fname based on time

### Other modifiable args
MODULE="${MODULE:-gpt_oss}"
CONFIG="${CONFIG:-gpt_oss_20b_pretrain_c4}"
TRAINING_STEPS="${TRAINING_STEPS:-15000}"
CONTAINER="${CONTAINER:-alto_gpt_oss}" # container name (for user readability)


# default Docker image from Han Wang
IMAGE="${IMAGE:-wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch}"

# -----------------------------------------------------------------------------
# Docker Setup
# -----------------------------------------------------------------------------

mkdir -p \
    "$HF_HOME_DIR" \
    "$CHECKPOINT_DIR" \
    "$(dirname "$LOG_FILE")"

echo "=== ALTO GPT-OSS 20B BF16 baseline ==="
echo "Node:           $(hostname)"
echo "Image:          $IMAGE"
echo "Config:         $CONFIG"
echo "GPUs:           $NGPU"
echo "Training steps: $TRAINING_STEPS"
echo "Model directory: $HF_HOME_DIR"
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
    --env-file "$HF_ENV_FILE"
    -v "$HOME:$HOME"
    -v "$ALTO_DIR:/alto"
    -v "$DATA_DIR:$DATA_DIR"
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

# Hardware resources are added only when they exist.
DEVICE_PATHS="${DEVICE_PATHS:-/dev/kfd /dev/dri /dev/infiniband}"
DEVICE_GROUPS="${DEVICE_GROUPS:-render video}"

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

# start docker contrainer
docker run "${docker_args[@]}" "$IMAGE" sleep infinity

# make sure docker is gracefully stopped quickly on exit
cleanup() {
    status=$?

    # Prevent cleanup from being triggered recursively.
    trap - EXIT INT TERM

    echo
    echo "[train] Stopping container $CONTAINER ..."

    docker stop --time 3 "$CONTAINER" >/dev/null 2>&1 ||
        docker kill "$CONTAINER" >/dev/null 2>&1 ||
        true

    exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# -----------------------------------------------------------------------------
# Load model and install additional docker dependencies
# -----------------------------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-$HF_HOME_DIR/models/gpt-oss-20b}"
echo "[model] Ensuring tokenizer is available at $MODEL_DIR ..."

if [[ -f "$MODEL_DIR/tokenizer.json" ]]; then
    echo "[model] Tokenizer already present, skipping download."
else
    echo "[model] Downloading tokenizer ..."
    docker exec "$CONTAINER" \
        hf download openai/gpt-oss-20b \
            --include "tokenizer*" "special_tokens_map.json" "config.json" \
            --local-dir "$MODEL_DIR"
fi


echo "[train] Installing dependencies ..."

docker exec "$CONTAINER" bash -c "
    python3 -m pip install -q torchao &&
    python3 -m pip install -q \
        --no-build-isolation \
        --no-deps \
        -e /alto/3rdparty/torchtitan
"

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
        --dump_folder "$CHECKPOINT_DIR" \
        --profiling.enable_profiling \
        --profiling.profile_freq 1000 \
        --profiling.profiler_warmup 3 \
        --profiling.profiler_active 1 \
    2>&1 | tee "$LOG_FILE"

echo "[train] Run complete."