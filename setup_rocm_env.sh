#!/bin/bash
# =============================================================================
# Setup a uv virtual environment with PyTorch 2.10 (ROCm 7.0) + Flash Attention
#
# System requirements:
#   - ROCm 7.0 installed at /opt/rocm-7.0.0
#   - AMD MI300X (gfx942) GPUs
#   - Python 3.12
#   - curl available
#
# Usage:
#   bash setup_rocm_env.sh          # create and install (~50 min, flash-attn build is slow)
#   source .venv/bin/activate       # activate the environment
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
ROCM_VERSION="7.0"
TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"

echo "============================================"
echo "  ROCm ${ROCM_VERSION} + PyTorch 2.10 Environment Setup"
echo "============================================"

# ---- Step 1: Install uv if not available ------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo "  uv installed: $(uv --version)"
else
    echo "[1/5] uv already available: $(uv --version)"
fi

# ---- Step 2: Create virtual environment -------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[2/5] Virtual environment already exists at $VENV_DIR"
else
    echo "[2/5] Creating virtual environment at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python 3.12
fi

# ---- Step 3: Install PyTorch 2.10 + ROCm 7.0 --------------------------------
echo "[3/5] Installing PyTorch 2.10 (ROCm ${ROCM_VERSION})..."
# NOTE: Must use explicit +rocm7.0 suffix to avoid pulling the CUDA wheel
uv pip install --python "$VENV_DIR/bin/python" \
    "torch==2.10.0+rocm7.0" \
    "torchvision==0.25.0+rocm7.0" \
    "torchaudio==2.10.0+rocm7.0" \
    --index-url "$TORCH_INDEX" \
    --reinstall

# ---- Step 4: Install Flash Attention (ROCm CK build from source) ------------
echo "[4/5] Installing Flash Attention for ROCm (building from source, ~45 min)..."
# flash-attn needs torch, wheel, setuptools, packaging, ninja, psutil at build time
uv pip install --python "$VENV_DIR/bin/python" \
    wheel setuptools packaging ninja psutil

MAX_JOBS="${MAX_JOBS:-16}" uv pip install --python "$VENV_DIR/bin/python" \
    flash-attn \
    --no-build-isolation

# ---- Step 5: Install project dependencies -----------------------------------
echo "[5/5] Installing project dependencies..."
uv pip install --python "$VENV_DIR/bin/python" \
    compressed-tensors \
    pydantic \
    tqdm \
    loguru \
    einops \
    tyro \
    transformers \
    meson-python meson pybind11

# Install torchtitan (editable) and Model-Optimizer
uv pip install --python "$VENV_DIR/bin/python" \
    --no-build-isolation -e "$SCRIPT_DIR/3rdparty/torchtitan"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Activate with: source $VENV_DIR/bin/activate"
echo "============================================"
echo ""

# ---- Verification -----------------------------------------------------------
echo "Running verification..."
"$VENV_DIR/bin/python" -c "
import torch
print('PyTorch:', torch.__version__, '| HIP:', torch.version.hip)
print('GPU:', torch.cuda.is_available(), f'({torch.cuda.device_count()} devices)')
import flash_attn
print('Flash Attention:', flash_attn.__version__)
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 128, 8, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, k, v)
print('Flash Attention forward pass:', out.shape, '- OK')
print('All checks passed!')
"
