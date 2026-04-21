#!/bin/bash

# Docker image for mxfp8 testing
IMAGE=ghcr.io/amd-agi/han-workspace:ubuntu22.04-pytorch2.12.0dev20260217-rocm7.2

# Get script directory (project root)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Get group IDs for GPU access
GID_RENDER=$(getent group render | cut -d: -f3)
GID_VIDEO=$(getent group video | cut -d: -f3)
CONTAINER_NAME=${MXFP8_CONTAINER_NAME:-node-exporter-mxfp8}
HOST_USER=$(id -un)

# Avoid name conflict with stale exited container.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run --rm -it -u $(id -u):$GID_RENDER \
    --name "$CONTAINER_NAME" \
    -e HOME=/tmp \
    -e USER="$HOST_USER" \
    -e PYTHONUSERBASE=/tmp/.local \
    -e PIP_CACHE_DIR=/tmp/.cache/pip \
    -e XDG_CACHE_HOME=/tmp/.cache \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/.cache/torchinductor \
    -e CC=gcc \
    -e CXX=g++ \
    --ulimit core=0 --privileged \
    --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add $GID_RENDER \
    --group-add $GID_VIDEO \
    --network host \
    --ipc=host --shm-size 8G \
    --workdir /workspace \
    -v "$SCRIPT_DIR:/workspace" \
    $IMAGE
