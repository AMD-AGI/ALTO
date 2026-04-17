#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="ghcr.io/amd-agi/han-workspace:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2"
CONTAINER_NAME="model-optimizer-rocm-shell"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
GID_RENDER=$(getent group render | cut -d: -f3 || true)
GID_VIDEO=$(getent group video | cut -d: -f3 || true)

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "Image '$IMAGE_NAME' not found locally, pulling..."
    docker pull "$IMAGE_NAME"
fi

docker_args=(
    --rm
    -it
    --name "$CONTAINER_NAME"
    --ulimit core=0
    --privileged
    --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined
    --device=/dev/kfd
    --device=/dev/dri
    --network=host
    --ipc=host
    --shm-size=16g
    --workdir /workspace/Model-Optimizer
    -v "$SCRIPT_DIR":/workspace/Model-Optimizer
)

if [[ -n "$GID_RENDER" ]]; then
    docker_args+=(--group-add "$GID_RENDER")
fi

if [[ -n "$GID_VIDEO" ]]; then
    docker_args+=(--group-add "$GID_VIDEO")
fi

docker run "${docker_args[@]}" \
    "$IMAGE_NAME" \
    bash
