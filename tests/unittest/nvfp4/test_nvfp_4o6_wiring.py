# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Wiring tests for Four Over Six (4/6) through the modifier + dispatch layers.

Covers:
  * modifier config fields (``use_4o6`` / ``select_metric``) and the MXFP4
    rejection validator (4/6 is NVFP4-only),
  * propagation of the fields into ``TrainingOpConfig``,
  * dispatch forwarding the fields into the dense NVFP4 linear kernel, and
  * an end-to-end NVFP4 linear forward/backward with 4/6 enabled, confirming
    4/6 perturbs the activation path while keeping gradients finite.
"""

import pytest
import torch

import alto.kernels.dispatch.tensor as dispatch_tensor
from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.dispatch.tensor import NVFP4TrainingWeightWrapperTensor
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import ALIGN_SIZE_M
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier


# ---------------------------------------------------------------------------
# Modifier-level: config fields + MXFP4 rejection (pure pydantic, no GPU).
# ---------------------------------------------------------------------------

def _resolved_configs(modifier):
    return list(modifier.resolved_config.keys())


def test_modifier_4o6_defaults_off():
    mod = LowPrecisionTrainingModifier(scheme="nvfp4")
    assert mod.use_4o6 is False
    assert mod.select_metric == "mae"
    cfg = _resolved_configs(mod)[0]
    assert cfg.use_4o6 is False
    assert cfg.select_metric == "mae"


def test_modifier_4o6_propagates_to_training_op_config():
    mod = LowPrecisionTrainingModifier(
        scheme="nvfp4", use_4o6=True, select_metric="mse")
    cfg = _resolved_configs(mod)[0]
    assert cfg.precision == "nvfp4"
    assert cfg.use_4o6 is True
    assert cfg.select_metric == "mse"


def test_modifier_rejects_4o6_with_mxfp4_scheme():
    with pytest.raises(ValueError, match="nvfp4"):
        LowPrecisionTrainingModifier(scheme="mxfp4", use_4o6=True)


def test_modifier_rejects_4o6_with_mxfp4_in_dict_scheme():
    with pytest.raises(ValueError, match="nvfp4"):
        LowPrecisionTrainingModifier(
            scheme={"nvfp4": ["Linear"], "mxfp4": ["re:.*experts"]}, use_4o6=True)


def test_modifier_allows_4o6_off_with_mxfp4():
    # MXFP4 without 4/6 must remain valid.
    mod = LowPrecisionTrainingModifier(scheme="mxfp4", use_4o6=False)
    assert mod.use_4o6 is False


# ---------------------------------------------------------------------------
# Dispatch-level: the dense NVFP4 linear path must forward 4/6 config.
# ---------------------------------------------------------------------------

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


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 dispatch/kernel requires a GPU"
)


@requires_gpu
@pytest.mark.parametrize("use_4o6,select_metric", [(True, "mse"), (False, "mae")])
def test_linear_routes_4o6_config_to_nvfp4_kernel(monkeypatch, use_4o6, select_metric):
    calls = []

    def _mock_linear(A, B, *, use_2dblock_x, use_2dblock_w, use_sr_grad,
                     use_outer_scale, use_hadamard, use_dge,
                     use_4o6, select_metric):
        calls.append({"use_4o6": use_4o6, "select_metric": select_metric})
        return A.new_full((A.shape[0], B.shape[0]), 5.0)

    monkeypatch.setattr(dispatch_tensor, "_to_nvfp4_then_scaled_mm", _mock_linear)

    cfg = _make_config(use_4o6=use_4o6, select_metric=select_metric)
    w = torch.randn(32, 16, dtype=torch.bfloat16, device="cuda")
    W_wrapped = NVFP4TrainingWeightWrapperTensor(w, cfg)
    x = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")

    torch.nn.functional.linear(x, W_wrapped)

    assert len(calls) == 1
    assert calls[0]["use_4o6"] is use_4o6
    assert calls[0]["select_metric"] == select_metric


# ---------------------------------------------------------------------------
# End-to-end: NVFP4 linear with 4/6 enabled (activations only).
# ---------------------------------------------------------------------------

@requires_gpu
def test_nvfp4_linear_4o6_forward_backward_runs():
    from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_scaled_mm

    torch.manual_seed(0)
    # near-maximal-value activations: the regime where 4/6 changes selection.
    x = (torch.rand(64, 256, device="cuda", dtype=torch.bfloat16) * 0.3 + 0.7)
    w = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)

    def run(use_4o6):
        xi = x.clone().requires_grad_(True)
        wi = w.clone().requires_grad_(True)
        y = _to_nvfp4_then_scaled_mm(
            xi, wi, use_2dblock_x=False, use_2dblock_w=True, use_sr_grad=False,
            use_outer_scale=True, use_hadamard=False, use_dge=False,
            use_4o6=use_4o6, select_metric="mae")
        y.sum().backward()
        return y.detach(), xi.grad.detach(), wi.grad.detach()

    y0, gx0, gw0 = run(False)
    y1, gx1, gw1 = run(True)

    for t in (y1, gx1, gw1):
        assert torch.isfinite(t).all()
    # 4/6 changes the activation quantization, so the forward output must move.
    assert not torch.allclose(y0, y1), "use_4o6=True should perturb the output"


# ---------------------------------------------------------------------------
# Grouped GEMM (MoE experts): the grouped path must forward + apply 4/6 too.
# 4/6 reaches expert ACTIVATION quant only (weights/gradients stay standard).
# ---------------------------------------------------------------------------

def _make_contiguous_expert_indices(M_total, num_groups, num_experts, device):
    indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        eid = int(torch.randint(0, num_experts, (1,), device=device).item())
        s = g * ALIGN_SIZE_M
        indices[s:s + ALIGN_SIZE_M] = eid
    return indices


@requires_gpu
@pytest.mark.parametrize("use_4o6,select_metric", [(True, "mse"), (False, "mae")])
def test_grouped_routes_4o6_config_to_nvfp4_kernel(monkeypatch, use_4o6, select_metric):
    """Dispatch must forward 4/6 config into the grouped (MoE) NVFP4 path."""
    calls = []

    def _mock_grouped(A, B, *, offs, use_2dblock_x, use_2dblock_w, use_sr_grad,
                      use_outer_scale, use_hadamard, use_dge,
                      use_4o6, select_metric):
        calls.append({"use_4o6": use_4o6, "select_metric": select_metric})
        return A.new_full((A.shape[0], B.shape[-1]), 7.0)

    monkeypatch.setattr(dispatch_tensor, "_quantize_then_nvfp4_scaled_grouped_mm", _mock_grouped)

    cfg = _make_config(use_4o6=use_4o6, select_metric=select_metric)
    num_experts, K, N, M = 2, 16, 32, ALIGN_SIZE_M
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    W_wrapped = NVFP4TrainingWeightWrapperTensor(
        torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda"), cfg)
    offs = torch.tensor([M // num_experts, M], dtype=torch.int32, device="cuda")

    torch._grouped_mm(A, W_wrapped, offs=offs)

    assert len(calls) == 1
    assert calls[0]["use_4o6"] is use_4o6
    assert calls[0]["select_metric"] == select_metric


@requires_gpu
def test_nvfp4_grouped_4o6_forward_backward_runs():
    """Grouped GEMM with 4/6 on expert activations runs and perturbs output."""
    from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import nvfp4_grouped_gemm

    torch.manual_seed(0)
    M_total, N, K, num_experts = 512, 256, 256, 4
    num_groups = M_total // ALIGN_SIZE_M
    # near-maximal-value activations: the regime where 4/6 changes selection.
    x = (torch.rand(M_total, K, device="cuda", dtype=torch.bfloat16) * 0.3 + 0.7)
    w = torch.randn(num_experts, N, K, device="cuda", dtype=torch.bfloat16)
    idx = _make_contiguous_expert_indices(M_total, num_groups, num_experts, "cuda")

    def run(use_4o6):
        xi = x.clone().requires_grad_(True)
        wi = w.clone().requires_grad_(True)
        y = nvfp4_grouped_gemm(
            xi, wi, idx, trans_weights=True,
            use_2dblock_x=False, use_2dblock_w=True, use_sr_grad=False,
            use_outer_scale=True, use_hadamard=False, use_dge=False,
            use_4o6=use_4o6, select_metric="mae")
        y.sum().backward()
        return y.detach(), xi.grad.detach(), wi.grad.detach()

    y0, gx0, gw0 = run(False)
    y1, gx1, gw1 = run(True)

    for t in (y1, gx1, gw1):
        assert torch.isfinite(t).all()
    assert not torch.allclose(y0, y1), "grouped use_4o6=True should perturb the output"


@requires_gpu
def test_nvfp4_grouped_4o6_default_off_is_unchanged():
    """grouped use_4o6=False must reproduce the standard grouped output exactly."""
    from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import nvfp4_grouped_gemm

    torch.manual_seed(1)
    M_total, N, K, num_experts = 256, 128, 128, 2
    num_groups = M_total // ALIGN_SIZE_M
    x = torch.randn(M_total, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(num_experts, N, K, device="cuda", dtype=torch.bfloat16)
    idx = _make_contiguous_expert_indices(M_total, num_groups, num_experts, "cuda")

    kw = dict(trans_weights=True, use_2dblock_x=False, use_2dblock_w=True,
              use_sr_grad=False, use_outer_scale=True, use_hadamard=False, use_dge=False)
    y_default = nvfp4_grouped_gemm(x, w, idx, **kw)
    y_off = nvfp4_grouped_gemm(x, w, idx, use_4o6=False, **kw)
    assert torch.equal(y_default, y_off)
