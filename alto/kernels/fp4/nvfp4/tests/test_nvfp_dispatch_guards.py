# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Dispatch-layer guard tests for the NVFP4 weight-wrapper tensor.

These tests pin down two behaviours that, if regressed, would cause a silent
downgrade to BF16 under a ``scheme=nvfp4`` configuration (see review items
B1 and B2):

* ``torch._grouped_mm`` on an NVFP4-wrapped weight must raise
  ``NotImplementedError``, not quietly fall through to the default BF16 kernel.
* ``F.linear`` / ``mm`` / ``addmm`` on an NVFP4-wrapped weight must refuse
  configs with ``use_hadamard=True`` or ``use_dge=True`` — neither is
  implemented on the NVFP4 linear path yet.
"""

import pytest
import torch

from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor


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
# B1 — grouped_mm must hard-fail rather than silently fall back to BF16
# ---------------------------------------------------------------------------

def test_grouped_mm_raises_not_implemented(device):
    """B1: applying scheme='nvfp4' to a module that issues torch._grouped_mm
    (e.g. MoE GroupedExperts) must raise, not silently run BF16."""
    cfg = _make_config()
    # 2D × 3D grouped-matmul signature (MoE routed-experts shape).
    num_experts, K, N, M = 2, 16, 16, 16
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device=device)
    W_wrapped = NVFP4TrainingWeightWrapperTensor(W, cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device=device)

    with pytest.raises(NotImplementedError, match="grouped"):
        torch._grouped_mm(A, W_wrapped, offs=offs)


# ---------------------------------------------------------------------------
# B2 — unsupported recipe knobs must hard-fail rather than be silently dropped
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["use_hadamard", "use_dge"])
def test_linear_rejects_unsupported_flag(device, flag):
    """B2: setting use_hadamard=True or use_dge=True on the NVFP4 scheme must
    raise at call time — otherwise the flag is silently no-op'd, meaning the
    user gets a plain NVFP4 path despite requesting Hadamard/DGE."""
    cfg = _make_config(**{flag: True})
    W_wrapped = _make_wrapper(cfg, device=device)
    x = torch.randn(16, 32, dtype=torch.bfloat16, device=device)

    with pytest.raises(AssertionError, match=flag):
        torch.nn.functional.linear(x, W_wrapped)


# ---------------------------------------------------------------------------
# Baseline sanity — valid configs still work after the new guards
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad",   [False, True])
def test_linear_supported_knobs_still_work(device, use_2dblock_x, use_2dblock_w, use_sr_grad):
    """Make sure the B1/B2 guards did not regress the legitimately supported
    knobs (2D block scaling on x/w and SR on grads)."""
    cfg = _make_config(
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
    )
    W_wrapped = _make_wrapper(cfg, device=device)
    x = torch.randn(16, 32, dtype=torch.bfloat16, device=device)
    y = torch.nn.functional.linear(x, W_wrapped)
    assert y.shape == (16, 32)
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
