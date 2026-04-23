#!/bin/bash
# =============================================================================
# Run NVFP4 quantization kernel tests in an isolated Docker container.
#
# Usage:
#   bash alto/kernels/fp4/nvfp4/tests/run_tests.sh
#
# Requirements:
#   - Docker with GPU access (--device=/dev/kfd, --device=/dev/dri)
#   - ROCm-compatible AMD GPU
#   - Docker image: rocm/pytorch:latest (or override via DOCKER_IMAGE env var)
#   - Local torchtitan source at TORCHTITAN_SRC (default: same-level checkout)
#
# The script:
#   1. Creates a Docker container with GPU access
#   2. Upgrades PyTorch to 2.10 (ROCm 7.0) for torch.nn.attention.varlen support
#   3. Installs torchtitan (from 3rdparty/torchtitan or TORCHTITAN_SRC)
#   4. Installs remaining dependencies
#   5. Runs the full nvfp4 test suite via standard package imports
#   6. Cleans up the container
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NVFP4_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$NVFP4_DIR/../../../.." && pwd)"

DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/pytorch:latest}"
CONTAINER_NAME="nvfp4-test-$$"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

# torchtitan source: prefer the initialized submodule, fall back to env var
TORCHTITAN_SRC="${TORCHTITAN_SRC:-${REPO_ROOT}/3rdparty/torchtitan}"
if [ ! -f "${TORCHTITAN_SRC}/pyproject.toml" ]; then
    echo "ERROR: torchtitan source not found at ${TORCHTITAN_SRC}"
    echo "       Either run 'git submodule update --init' or set TORCHTITAN_SRC."
    exit 1
fi

# Expected torchtitan commit (from .gitmodules)
TORCHTITAN_COMMIT="${TORCHTITAN_COMMIT:-27960210f3053f38de77f0d2d07c9f1485ed246d}"

TORCH_INDEX="https://download.pytorch.org/whl/rocm7.0"

echo "============================================"
echo "  NVFP4 Kernel Test Runner"
echo "============================================"
echo "  Docker image    : ${DOCKER_IMAGE}"
echo "  Container       : ${CONTAINER_NAME}"
echo "  GPU devices     : ${HIP_VISIBLE_DEVICES}"
echo "  Repo root       : ${REPO_ROOT}"
echo "  torchtitan src  : ${TORCHTITAN_SRC}"
echo "============================================"
echo ""

cleanup() {
    echo ""
    echo "[cleanup] Removing container ${CONTAINER_NAME} ..."
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --- Step 1: Create container ------------------------------------------------
echo "[1/5] Creating Docker container ..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --group-add render \
    --ipc=host \
    -v "${REPO_ROOT}:/workspace/agi-model-opt:ro" \
    -v "${TORCHTITAN_SRC}:/workspace/torchtitan-src:ro" \
    "${DOCKER_IMAGE}" \
    sleep infinity >/dev/null

# --- Step 2: Upgrade PyTorch to 2.10 ----------------------------------------
echo "[2/5] Upgrading PyTorch to 2.10 (ROCm 7.0) ..."
docker exec "${CONTAINER_NAME}" \
    pip install -q "torch==2.10.0+rocm7.0" "torchvision==0.25.0+rocm7.0" \
    --index-url "${TORCH_INDEX}"

# --- Step 3: Install torchtitan ---------------------------------------------
echo "[3/5] Installing torchtitan ..."
docker exec "${CONTAINER_NAME}" bash -c "
    cp -r /workspace/torchtitan-src /tmp/torchtitan-build
    cd /tmp/torchtitan-build
    git checkout -f ${TORCHTITAN_COMMIT} 2>/dev/null || true
    pip install -q meson-python meson pybind11
    pip install -q --no-build-isolation -e /tmp/torchtitan-build
"

# --- Step 4: Install remaining dependencies ---------------------------------
echo "[4/5] Installing dependencies ..."
docker exec "${CONTAINER_NAME}" \
    pip install -q safetensors compressed-tensors pydantic loguru psutil pytest triton

# --- Step 5: Run tests -------------------------------------------------------
echo "[5/5] Running tests ..."
echo ""

docker exec \
    -e PYTHONPATH=/workspace/agi-model-opt \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
    -e TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \
    "${CONTAINER_NAME}" \
    python3 -m pytest \
        /workspace/agi-model-opt/alto/kernels/fp4/nvfp4/tests/test_nvfp_quantization.py \
        -v --tb=short

echo ""
echo "============================================"
echo "  All tests passed."
echo "============================================"
