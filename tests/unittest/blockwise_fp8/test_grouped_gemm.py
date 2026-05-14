# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from tabulate import tabulate

from alto.kernels.blockwise_fp8.grouped_gemm import ALIGN_SIZE_M, cg_grouped_gemm
from alto.kernels.blockwise_fp8.grouped_gemm.cg_backward import verify_cg_gemm_backward


def _build_expert_indices(m_total: int, num_experts: int, device: torch.device) -> torch.Tensor:
    num_groups = m_total // ALIGN_SIZE_M
    indices = torch.zeros(m_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        expert_idx = torch.randint(0, num_experts, (1,), device=device, dtype=torch.int32).item()
        start = g * ALIGN_SIZE_M
        end = (g + 1) * ALIGN_SIZE_M
        indices[start:end] = expert_idx
    return indices


def _reference_grouped_gemm(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    trans_weights: bool,
) -> torch.Tensor:
    m_total = expert_indices.shape[0]
    n = expert_weights.shape[-2] if trans_weights else expert_weights.shape[-1]
    out = torch.zeros((m_total, n), dtype=inputs.dtype, device=inputs.device)
    num_groups = m_total // ALIGN_SIZE_M
    for g in range(num_groups):
        start = g * ALIGN_SIZE_M
        end = (g + 1) * ALIGN_SIZE_M
        expert_idx = expert_indices[start].item()
        w = expert_weights[expert_idx]
        if trans_weights:
            w = w.t()
        out[start:end] = inputs[start:end] @ w
    return out


def _calc_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_f = x.float()
    y_f = y.float()
    signal = torch.sum(x_f**2)
    noise = torch.sum((x_f - y_f)**2)
    return (10.0 * torch.log10(signal / (noise + 1e-12))).item()


def _calc_cossim(x: torch.Tensor, y: torch.Tensor) -> float:
    x_f = x.float().reshape(-1)
    y_f = y.float().reshape(-1)
    return (torch.sum(x_f * y_f) / (torch.norm(x_f) * torch.norm(y_f) + 1e-12)).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton grouped GEMM tests")
@pytest.mark.parametrize("use_fp8", [False, True])
def test_cg_backward_verify_path_small_shape(use_fp8: bool):
    # Cover the for-loop reference path used in cg_backward.py's __main__ example.
    inputs_ok, weights_ok = verify_cg_gemm_backward(
        M_total=ALIGN_SIZE_M * 2,
        N=128,
        K=128,
        num_experts=4,
        device="cuda",
        atol=2e-1,
        rtol=2e-1,
        use_fp8=use_fp8,
    )
    assert inputs_ok
    assert weights_ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton grouped GEMM tests")
@pytest.mark.parametrize("use_fp8", [False, True])
@pytest.mark.parametrize("trans_weights", [True, False])
@pytest.mark.parametrize("non_contiguous", [False, True])
def test_cg_grouped_gemm_contiguous_and_non_contiguous(use_fp8: bool, trans_weights: bool, non_contiguous: bool):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    m_total = ALIGN_SIZE_M * 2
    n = 128
    k = 128
    num_experts = 4

    inputs_base = torch.randn((m_total, k), device=device, dtype=dtype)
    if trans_weights:
        weights_base = torch.randn((num_experts, n, k), device=device, dtype=dtype)
    else:
        weights_base = torch.randn((num_experts, k, n), device=device, dtype=dtype)

    if non_contiguous:
        inputs_data = inputs_base.transpose(-1, -2).contiguous().transpose(-1, -2)
        weights_data = weights_base.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not inputs_data.is_contiguous()
        assert not weights_data.is_contiguous()
    else:
        inputs_data = inputs_base.contiguous()
        weights_data = weights_base.contiguous()
        assert inputs_data.is_contiguous()
        assert weights_data.is_contiguous()

    expert_indices = _build_expert_indices(m_total, num_experts, device)
    target = torch.randn((m_total, n), device=device, dtype=dtype)

    inputs_ref = torch.nn.Parameter(inputs_data.clone())
    weights_ref = torch.nn.Parameter(weights_data.clone())
    ref_out = _reference_grouped_gemm(inputs_ref, weights_ref, expert_indices, trans_weights=trans_weights)
    ref_loss = torch.nn.functional.mse_loss(ref_out, target)
    ref_loss.backward()
    ref_dx = inputs_ref.grad.detach().clone()
    ref_dw = weights_ref.grad.detach().clone()

    inputs_test = torch.nn.Parameter(inputs_data.clone())
    weights_test = torch.nn.Parameter(weights_data.clone())
    test_out = cg_grouped_gemm(
        inputs_test,
        weights_test,
        expert_indices,
        use_fp8=use_fp8,
        trans_weights=trans_weights,
    )
    test_loss = torch.nn.functional.mse_loss(test_out, target)
    test_loss.backward()

    out_snr = _calc_snr(test_out, ref_out)
    dx_snr = _calc_snr(inputs_test.grad, ref_dx)
    dw_snr = _calc_snr(weights_test.grad, ref_dw)
    out_sim = _calc_cossim(test_out, ref_out)
    dx_sim = _calc_cossim(inputs_test.grad, ref_dx)
    dw_sim = _calc_cossim(weights_test.grad, ref_dw)

    print()
    print(
        tabulate(
            [
                ["O", out_snr, out_sim],
                ["dX", dx_snr, dx_sim],
                ["dW", dw_snr, dw_sim],
            ],
            headers=["Tensor", "SNR", "Cosine Sim"],
            tablefmt="github",
        ))

    assert out_snr > 25.0
    assert dx_snr > 25.0
    assert dw_snr > 25.0
