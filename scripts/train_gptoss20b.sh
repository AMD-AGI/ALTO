#!/usr/bin/env bash
# Run ALTO GPT-OSS 20B training on the current node.
# Supports Docker and Singularity/Apptainer — auto-detected, or set RUNTIME=singularity|docker.
#
# Example:
#   NGPU=4 CONFIG=gpt_oss_debugmodel TRAINING_STEPS=20 \
#       bash ~/ALTO/scripts/train_gptoss20b.sh
#
#
# ######## STEP 0: Install ALTO repository and update ALTO_DIR below
# git clone --recurse-submodules https://github.com/AMD-AGI/ALTO.git
#
#
# ######## STEP 1: Download the C4 dataset
# ######## make sure to update config_registry.py with appropriate data location
# ###  OPTION 1:
# # Create desired download directory with the right permission
# # cd /data/gpt_oss_20b
# # Download training and validation data
# # bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
# #     -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri
# ### OPTION 2:
# # C4_CACHE="$HF_HOME_SHARED/datasets/allenai___c4"
# # if [ -d "$C4_CACHE" ] && [ -n "$(ls -A "$C4_CACHE" 2>/dev/null)" ]; then
# #     echo "[train] C4 dataset already cached, skipping download."
# # else
# #     echo "[train] Downloading C4 dataset (this may take a while) ..."
# #     docker exec "$CONTAINER" bash -c "
# #         python3 -c \"
# # from datasets import load_dataset
# # load_dataset('allenai/c4', 'en', split='train')
# # load_dataset('allenai/c4', 'en', split='validation')
# #         \"
# #     "
# # fi
#
# ####### Viewing Loss Curves
# # tensorboard events are saved in the checkpointing directory, one can
# # view these by using the following command:
# # tensorboard --logdir $CHECKPOINT_DIR --host 127.0.0.1 --port 6006
# #
# # If running on remote machine, you will want to forward the port to the local machine:
# # ssh -L 6006:localhost:6006 nfrumkin@useocpslog-002

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SLURM_JOB_ID="${SLURM_JOB_ID:-${SLURM_JOBID:-}}"
if [[ -n "$SLURM_JOB_ID" ]]; then
    RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
else
    RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
fi

### Machine-specific args
NGPU="${NGPU:-8}"
PROF_FREQ="${PROF_FREQ:-100000}"
HF_HOME_DIR="${HF_HOME_DIR:-/shared_inference/alirezak/hf_home}"
DATA_DIR="${DATA_DIR:-/shared_inference/alirezak/hf_home/data}"
HF_ENV_FILE="${HF_ENV_FILE:-$HOME/.hf.env}"

### Run-specific args
ALTO_DIR="${ALTO_DIR:-$PWD}"
CONFIG="${CONFIG:-gpt_oss_20b_mxfp4_base}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/shared_inference/alirezak/gptoss_chkpt/${USER}/${CONFIG}_$RUN_ID}"
LOG_FILE="${LOG_FILE:-$ALTO_DIR/logs/${CONFIG}_$(date +%Y%m%d_%H%M%S).log}"

### Other modifiable args
MODULE="${MODULE:-gpt_oss}"
TRAINING_STEPS="${TRAINING_STEPS:-15000}"
EXTRA_ARGS="${EXTRA_ARGS:-}" # extra torchrun args, e.g. set by smoke test wrapper

### Container image
IMAGE="${IMAGE:-wanghanthu/torchtitan:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2-patch}"
# Path where the Singularity .sif is stored (or will be built)
SIF="${SIF:-$HOME/alto.sif}"

# -----------------------------------------------------------------------------
# Runtime detection
# -----------------------------------------------------------------------------

if [[ -n "${RUNTIME:-}" ]]; then
    _runtime="$RUNTIME"
elif command -v docker &>/dev/null; then
    _runtime=docker
elif command -v apptainer &>/dev/null; then
    _runtime=apptainer
elif command -v singularity &>/dev/null; then
    _runtime=singularity
else
    echo "ERROR: no container runtime found (docker / singularity / apptainer)" >&2
    exit 1
fi

CONTAINER="${CONTAINER:-${CONFIG}_${RUN_ID}}"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

MODEL_DIR="/hf_home/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
MODEL_DIR_HOST="${HF_HOME_DIR}/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"

echo "=== ALTO GPT-OSS 20B ==="
echo "Node:           $(hostname)"
[[ -n "$SLURM_JOB_ID" ]] && echo "SLURM job:      $SLURM_JOB_ID"
echo "Runtime:        $_runtime"
echo "Image:          $IMAGE"
echo "Config:         $CONFIG"
echo "GPUs:           $NGPU"
echo "Training steps: $TRAINING_STEPS"
echo "Model directory: $HF_HOME_DIR"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "Log:            $LOG_FILE"
echo

mkdir -p "$HF_HOME_DIR" "$CHECKPOINT_DIR" "$(dirname "$LOG_FILE")" || true

# -----------------------------------------------------------------------------
# Docker path
# -----------------------------------------------------------------------------

if [[ "$_runtime" == "docker" ]]; then

    docker pull "$IMAGE"

    docker_args=(
        -d --rm
        --name "$CONTAINER"
        --privileged
        --user "$(id -u):$(id -g)"
        --network host --ipc host
        --cap-add SYS_PTRACE
        --shm-size 512G
        --security-opt seccomp=unconfined
        --env-file "$HF_ENV_FILE"
        -v "$HOME:$HOME"
        -v "$ALTO_DIR:/alto"
        -v "$DATA_DIR:/data"
        -v "$HF_HOME_DIR:/hf_home"
        -v "$CHECKPOINT_DIR:/checkpoints"
        -v /etc/passwd:/etc/passwd:ro
        -v /etc/group:/etc/group:ro
        -e HOME="$HOME" -e USER="$(id -un)"
        -e HF_HOME=/hf_home
        -e HF_DATASETS_CACHE=/hf_home/datasets
        -e TRITON_CACHE_DIR=/tmp/triton_cache
        -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
        -e PYTHONNOUSERSITE=1
    )

    DEVICE_PATHS="${DEVICE_PATHS:-/dev/kfd /dev/dri /dev/infiniband}"
    DEVICE_GROUPS="${DEVICE_GROUPS:-render video}"
    for device in $DEVICE_PATHS; do
        [[ -e "$device" ]] && docker_args+=(--device "$device")
    done
    for group in $DEVICE_GROUPS; do
        gid="$(getent group "$group" | cut -d: -f3 || true)"
        [[ -n "$gid" ]] && docker_args+=(--group-add "$gid")
    done

    docker run "${docker_args[@]}" "$IMAGE" sleep infinity

    cleanup() {
        local status=$?
        trap - EXIT INT TERM
        echo; echo "[train] Stopping container $CONTAINER ..."
        docker stop --time 3 "$CONTAINER" >/dev/null 2>&1 ||
            docker kill "$CONTAINER" >/dev/null 2>&1 || true
        exit "$status"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    echo "[model] Ensuring tokenizer is available at $MODEL_DIR_HOST ..."
    if [[ -f "$MODEL_DIR_HOST/tokenizer.json" ]]; then
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
        python3 -m pip install -q 'torchao==0.16.0' &&
        python3 -m pip install -q --no-build-isolation --no-deps -e /alto/3rdparty/torchtitan
    "

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
            --profiling.enable_profiling \
            --profiling.profile_freq "$PROF_FREQ" \
            --profiling.profiler_warmup 3 \
            --profiling.profiler_active 1 \
            --module "$MODULE" \
            --config "$CONFIG" \
            --training.steps "$TRAINING_STEPS" \
            --comm.init_timeout_seconds 1800 \
            --hf_assets_path "$MODEL_DIR" \
            --dump_folder /checkpoints \
            $EXTRA_ARGS

# -----------------------------------------------------------------------------
# Singularity / Apptainer path
# -----------------------------------------------------------------------------

else

    # Build .sif from Docker Hub if not already present
    if [[ ! -f "$SIF" ]]; then
        echo "[sif] Building $SIF from docker://$IMAGE ..."
        "$_runtime" pull "$SIF" "docker://$IMAGE"
    fi

    sif_args=(
        --rocm
        --bind "$ALTO_DIR:/alto"
        --bind "$DATA_DIR:/data"
        --bind "$HF_HOME_DIR:/hf_home"
        --bind "$CHECKPOINT_DIR:/checkpoints"
        --env HF_HOME=/hf_home
        --env HF_DATASETS_CACHE=/hf_home/datasets
        --env TRITON_CACHE_DIR=/tmp/triton_cache
        --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache
        --env PYTORCH_ALLOC_CONF=expandable_segments:True
        --env PYTHONUNBUFFERED=1
    )
    [[ -f "$HF_ENV_FILE" ]] && sif_args+=(--env-file "$HF_ENV_FILE")
    # TRANSFORMERS_OFFLINE=1 only for training — not for the tokenizer download step
    sif_train_args=("${sif_args[@]}" --env TRANSFORMERS_OFFLINE=1)

    echo "[model] Ensuring tokenizer is available at $MODEL_DIR_HOST ..."
    if [[ ! -f "$MODEL_DIR_HOST/tokenizer.json" ]]; then
        echo "[model] Downloading tokenizer ..."
        "$_runtime" exec "${sif_args[@]}" "$SIF" \
            hf download openai/gpt-oss-20b \
                --include "tokenizer*" "special_tokens_map.json" "config.json" \
                --local-dir "$MODEL_DIR"
    else
        echo "[model] Tokenizer already present, skipping download."
    fi

    echo "[train] Installing dependencies ..."
    "$_runtime" exec "${sif_train_args[@]}" "$SIF" bash -c "
        python3 -m pip install -q 'torchao==0.16.0' &&
        python3 -m pip install -q --no-build-isolation --no-deps -e /alto/3rdparty/torchtitan
    "

    echo "[train] Launching $CONFIG for $TRAINING_STEPS steps on $NGPU GPUs ..."
    "$_runtime" exec "${sif_train_args[@]}" --pwd /alto "$SIF" \
        torchrun \
            --standalone \
            --nproc_per_node "$NGPU" \
            --local-ranks-filter 0 \
            --tee 3 \
            -m alto.train \
            --profiling.enable_profiling \
            --profiling.profile_freq "$PROF_FREQ" \
            --profiling.profiler_warmup 3 \
            --profiling.profiler_active 1 \
            --module "$MODULE" \
            --config "$CONFIG" \
            --training.steps "$TRAINING_STEPS" \
            --comm.init_timeout_seconds 1800 \
            --hf_assets_path "$MODEL_DIR" \
            --dump_folder /checkpoints \
            $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE"

fi

echo "[train] Run complete."
