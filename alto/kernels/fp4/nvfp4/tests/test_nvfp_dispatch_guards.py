# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Dispatch-layer guard tests for the NVFP4 weight-wrapper tensor.

These tests pin down behaviour that would otherwise silently change the active
low-precision recipe under ``scheme=nvfp4``.  The op-level tests verify the
math of ``NVFP4LinearFunction`` and ``nvfp4_grouped_gemm``; this file verifies
that the production wrapper / dispatch path preserves the user's config and
still routes gradients to the wrapper leaf.
"""

import pytest
import torch

from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import ALIGN_SIZE_M


def _make_config(**overrides) -> TrainingOpConfig:
    defaults = dict(
        precision="nvfp4",
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
    )
    defaults.update(overrides)
    return TrainingOpConfig(**defaults)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("NVFP4 dispatch guards require a CUDA device")
    return torch.device("cuda")


def _make_wrapper(config: TrainingOpConfig, shape=(32, 32), device="cuda"):
    w = torch.randn(*shape, dtype=torch.bfloat16, device=device)
    return NVFP4TrainingWeightWrapperTensor(w, config)


# ---------------------------------------------------------------------------
# grouped_mm smoke — scheme='nvfp4' on a 2D × 3D grouped_mm (MoE routed
# experts layout) must reach the NVFP4 grouped-GEMM path, produce a BF16
# output with the expected shape, and actually execute a matmul (not a silent
# all-zero fallback on non-native platforms).
# ---------------------------------------------------------------------------

def test_grouped_mm_smoke(device):
    cfg = _make_config()
    num_experts, K, N, M = 2, 16, 16, ALIGN_SIZE_M
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device=device)

    y = torch._grouped_mm(A, W_wrapped, offs=offs)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    assert y.abs().max().item() > 0, "grouped_mm smoke must exercise a real matmul"


@pytest.mark.parametrize("use_hadamard,use_dge", [
    (True, False),
    (False, True),
    (True, True),
])
def test_grouped_mm_recipe_surface_smoke(device, use_hadamard, use_dge):
    """Grouped-GEMM must accept the same recipe knobs as the dense NVFP4 linear
    path once the implementation is present.  This is a lightweight smoke test:
    grouped recipe variants should produce a BF16 output with the expected
    shape and should not silently fall back to BF16 or zero-output behaviour."""
    cfg = _make_config(
        use_2dblock_x=False,  # Hadamard is defined only on the 1D-x path
        use_2dblock_w=True,
        use_sr_grad=True,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
    num_experts, K, N, M = 2, 16, 16, ALIGN_SIZE_M
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device=device)

    y = torch._grouped_mm(A, W_wrapped, offs=offs)
    assert y.shape == (M, N)
    assert y.dtype == torch.bfloat16
    assert y.abs().max().item() > 0


# ---------------------------------------------------------------------------
# Baseline sanity — every supported recipe knob combination must produce a
# valid BF16 output of the right shape.  ``use_hadamard`` and ``use_dge`` used
# to hard-fail here (B2 guard) before Hadamard/DGE were implemented on the
# NVFP4 linear path — that guard is now replaced by functional coverage.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad",   [False, True])
@pytest.mark.parametrize("use_hadamard",  [False, True])
@pytest.mark.parametrize("use_dge",       [False, True])
def test_linear_supported_knobs_still_work(
    device, use_2dblock_x, use_2dblock_w, use_sr_grad, use_hadamard, use_dge,
):
    """Functional smoke: every (2D×/2Dw/SR/Hadamard/DGE) combination must
    return a BF16 output with the expected shape.  M is picked large enough
    (64) to satisfy the Hadamard block size (32 by default)."""
    cfg = _make_config(
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
    W_wrapped = _make_wrapper(cfg, device=device, shape=(32, 32))
    x = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    y = torch.nn.functional.linear(x, W_wrapped)
    assert y.shape == (64, 32)
    assert y.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# B3 — wrapper must flow THROUGH to NVFP4LinearFunction (no premature unwrap
# in the dispatch layer, to stay consistent with the MXFP4 pattern and to
# keep FSDP2 / __torch_dispatch__ observability hooks intact).
# ---------------------------------------------------------------------------

def test_linear_passes_wrapper_through_to_autograd_function(device):
    """B3: dispatch layer must pass ``B`` (wrapper) into NVFP4LinearFunction.apply
    rather than pre-unwrapping to ``B._data``.

    CRITICAL: Under a production ``swap_params`` wrap, ``module.weight.data``
    is detached (``requires_grad=False``), so ``wrapper._data.requires_grad``
    is False.  If the dispatch layer pre-unwraps to ``B._data`` before handing
    the tensor to the autograd function, PyTorch sees an input with
    ``requires_grad=False`` and silently drops the gradient — ``param.grad``
    stays ``None`` for the lifetime of training, and every wrapped Linear is
    frozen while the loss curve still looks superficially reasonable (BF16
    tail / embedding / output keep training).

    This test exactly mirrors the swap done by the production
    ``LowPrecisionTrainingModifier`` and asserts that:
      1. ``param.grad`` lands on the wrapper (not ``param._data.grad``), and
      2. one optimizer step actually moves the underlying weight data.
    """
    cfg = _make_config()
    lin = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device)
    # Mirror _wrap_linear_weights_with_nvfp4 / swap_params:
    #   detach .data → wrap → re-assign as Parameter with requires_grad=True.
    orig = lin.weight.data
    wrapped = NVFP4TrainingWeightWrapperTensor(orig, cfg)
    lin.weight = torch.nn.Parameter(wrapped, requires_grad=True)
    assert lin.weight._data.requires_grad is False, (
        "precondition: production swap path detaches the weight data"
    )

    x = torch.randn(16, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = lin(x)
    y.sum().backward()
    assert y.shape == (16, 32)
    assert y.dtype == torch.bfloat16
    assert x.grad is not None and x.grad.shape == x.shape

    # Grad MUST land on the wrapper's ``.grad`` slot, not on
    # ``param._data.grad``.  This is the regression guard for the
    # "weight frozen" footgun described in the docstring above.
    assert lin.weight.grad is not None, (
        "param.grad is None: autograd dropped the gradient because the "
        "dispatch layer pre-unwrapped to B._data (requires_grad=False) "
        "— optimizer will silently skip this parameter."
    )
    assert lin.weight._data.grad is None, (
        "param._data.grad should be None when the wrapper is the graph leaf."
    )

    # One optimizer step must actually move the weight.
    w_before = lin.weight._data.detach().clone()
    torch.optim.AdamW(lin.parameters(), lr=1e-1).step()
    w_after = lin.weight._data.detach()
    diff = (w_after.float() - w_before.float()).norm().item()
    assert diff > 1e-6, f"weight did not update (L2 diff = {diff})"


@pytest.mark.parametrize("use_hadamard,use_dge", [
    (True,  False),
    (False, True),
    (True,  True),
])
def test_hadamard_and_dge_flow_grads_through_wrapper(device, use_hadamard, use_dge):
    """Protocol guard: when Hadamard / DGE are enabled, the autograd tape must
    still put ``param.grad`` on the wrapper (not on ``param._data.grad``), and
    one optimizer step must actually move the weight.

    This is the same pattern as
    ``test_linear_passes_wrapper_through_to_autograd_function`` but parameterised
    over the two newly-enabled recipe knobs; the intent is to catch any future
    refactor that re-introduces pre-unwrapping specifically along the Hadamard
    / DGE paths."""
    cfg = _make_config(use_hadamard=use_hadamard, use_dge=use_dge)
    # M >= 32 for HadamardFactory's default block_size=32.
    lin = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16, device=device)
    orig = lin.weight.data
    lin.weight = torch.nn.Parameter(
        NVFP4TrainingWeightWrapperTensor(orig, cfg), requires_grad=True,
    )
    x = torch.randn(64, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = lin(x); y.sum().backward()
    assert lin.weight.grad       is not None, "param.grad is None with Hadamard/DGE"
    assert lin.weight._data.grad is None,     "_data.grad must be None"
    assert x.grad                is not None

    w_before = lin.weight._data.detach().clone()
    torch.optim.AdamW(lin.parameters(), lr=1e-1).step()
    diff = (lin.weight._data.float() - w_before.float()).norm().item()
    assert diff > 1e-6, f"weight did not update (L2 diff = {diff})"


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="needs torch.compile")
def test_linear_compiles_with_wrapper(device):
    """Smoke: torch.compile must be able to trace an nn.Linear whose weight
    is an NVFP4TrainingWeightWrapperTensor, given ``NVFP4LinearFunction`` is
    marked ``@torch.compiler.allow_in_graph``."""
    cfg = _make_config()

    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            w = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
            self.w = torch.nn.Parameter(
                NVFP4TrainingWeightWrapperTensor(w, cfg),
                requires_grad=False,
            )

        def forward(self, x):
            return torch.nn.functional.linear(x, self.w)

    model = torch.compile(_Mod().to(device))
    x = torch.randn(16, 32, dtype=torch.bfloat16, device=device)
    y = model(x)
    assert y.shape == (16, 32)
    assert y.dtype == torch.bfloat16
