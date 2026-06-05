# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.kernels.mxfp8.mxfp8_grouped_gemm import mxfp8_grouped_gemm
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_backward import (
    mxfp8_grouped_gemm_backward_inputs,
    mxfp8_grouped_gemm_backward_weights,
)
from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT

from .utils import prepare_data, convert_from_mxfp8_pytorch


def _cossim(x, y):
    return torch.nn.functional.cosine_similarity(x.flatten().float(), y.flatten().float(), dim=0).item()


def _make_indices(num_groups, num_experts, device):
    indices = torch.empty(num_groups * ALIGN_SIZE_M, dtype=torch.int32, device=device)
    for g in range(num_groups):
        indices[g * ALIGN_SIZE_M:(g + 1) * ALIGN_SIZE_M] = g % num_experts
    return indices


def _reference(inputs, expert_weights, indices, trans_weights):
    m_total = inputs.shape[0]
    n_dim = expert_weights.shape[1] if trans_weights else expert_weights.shape[2]
    out = torch.zeros((m_total, n_dim), dtype=inputs.dtype, device=inputs.device)
    for start in range(0, m_total, ALIGN_SIZE_M):
        expert_idx = indices[start].item()
        weight = expert_weights[expert_idx].t() if trans_weights else expert_weights[expert_idx]
        out[start:start + ALIGN_SIZE_M] = inputs[start:start + ALIGN_SIZE_M] @ weight
    return out


def _reference_dgrad(grad_output, expert_weights, indices, trans_weights):
    m_total = grad_output.shape[0]
    k_dim = expert_weights.shape[2] if trans_weights else expert_weights.shape[1]
    grad_inputs = torch.zeros((m_total, k_dim), dtype=grad_output.dtype, device=grad_output.device)
    for start in range(0, m_total, ALIGN_SIZE_M):
        end = start + ALIGN_SIZE_M
        expert_idx = indices[start].item()
        weight = expert_weights[expert_idx] if trans_weights else expert_weights[expert_idx].t()
        grad_inputs[start:end] = grad_output[start:end] @ weight
    return grad_inputs


def _reference_wgrad(grad_output, inputs, indices, num_experts, trans_weights):
    n_dim = grad_output.shape[1]
    k_dim = inputs.shape[1]
    grad_weights = torch.zeros(
        (num_experts, n_dim, k_dim) if trans_weights else (num_experts, k_dim, n_dim),
        dtype=grad_output.dtype,
        device=grad_output.device,
    )
    for start in range(0, inputs.shape[0], ALIGN_SIZE_M):
        end = start + ALIGN_SIZE_M
        expert_idx = indices[start].item()
        if trans_weights:
            grad_weights[expert_idx] += grad_output[start:end].t() @ inputs[start:end]
        else:
            grad_weights[expert_idx] += inputs[start:end].t() @ grad_output[start:end]
    return grad_weights


@pytest.mark.parametrize("trans_weights", [True, False])
@pytest.mark.parametrize("use_2dblock_go", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
def test_backward_inputs_matches_dequant_reference(trans_weights, use_2dblock_go, use_2dblock_w):
    m_total, n_dim, k_dim, num_experts = 384, 128, 128, 2
    dtype = torch.bfloat16

    grad_output = prepare_data((m_total, n_dim), dtype)
    weight_shape = (num_experts, n_dim, k_dim) if trans_weights else (num_experts, k_dim, n_dim)
    expert_weights = prepare_data(weight_shape, dtype)
    expert_indices = _make_indices(m_total // ALIGN_SIZE_M, num_experts, torch.device("cuda"))

    weight_axis = (-1 if trans_weights else -2) if use_2dblock_w else (-2 if trans_weights else -1)
    grad_output_lp, grad_output_scales = torch.ops.alto.convert_to_mxfp8(
        grad_output,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format="e4m3",
        axis=-1,
        is_2d_block=use_2dblock_go,
    )
    expert_weights_lp, expert_weight_scales = torch.ops.alto.convert_to_mxfp8(
        expert_weights,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format="e4m3",
        axis=weight_axis,
        is_2d_block=use_2dblock_w,
    )

    grad_inputs = mxfp8_grouped_gemm_backward_inputs(
        grad_output_lp,
        expert_weights_lp,
        expert_indices,
        grad_output_scales,
        expert_weight_scales,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_go,
        use_2dblock_w=use_2dblock_w,
        output_dtype=dtype,
    )

    grad_output_dq = convert_from_mxfp8_pytorch(
        grad_output_lp, grad_output_scales, torch.float32, BLOCK_SIZE_DEFAULT, -1, use_2dblock_go)
    expert_weights_dq = convert_from_mxfp8_pytorch(
        expert_weights_lp, expert_weight_scales, torch.float32, BLOCK_SIZE_DEFAULT, weight_axis, use_2dblock_w)
    grad_inputs_ref = _reference_dgrad(grad_output_dq, expert_weights_dq, expert_indices, trans_weights)

    cos = _cossim(grad_inputs, grad_inputs_ref)
    assert cos > 0.999, \
        f"dgrad kernel vs dequant-matmul cos-sim too low: {cos} (2d_go={use_2dblock_go}, 2d_w={use_2dblock_w})"


@pytest.mark.parametrize("trans_weights", [True, False])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
def test_backward_weights_matches_dequant_reference(trans_weights, use_2dblock_x):
    m_total, n_dim, k_dim, num_experts = 384, 128, 128, 2
    dtype = torch.bfloat16

    inputs = prepare_data((m_total, k_dim), dtype)
    grad_output = prepare_data((m_total, n_dim), dtype)
    expert_indices = _make_indices(m_total // ALIGN_SIZE_M, num_experts, torch.device("cuda"))

    grad_axis = -1 if use_2dblock_x else 0
    input_axis = -1 if use_2dblock_x else 0
    grad_output_lp, grad_output_scales = torch.ops.alto.convert_to_mxfp8(
        grad_output,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format="e4m3",
        axis=grad_axis,
        is_2d_block=use_2dblock_x,
    )
    inputs_lp, input_scales = torch.ops.alto.convert_to_mxfp8(
        inputs,
        block_size=BLOCK_SIZE_DEFAULT,
        mxfp_format="e4m3",
        axis=input_axis,
        is_2d_block=use_2dblock_x,
    )

    grad_weights = mxfp8_grouped_gemm_backward_weights(
        grad_output_lp,
        inputs_lp,
        expert_indices,
        num_experts,
        grad_output_scales,
        input_scales,
        trans_weights=trans_weights,
        use_2dblock_go=use_2dblock_x,
        use_2dblock_x=use_2dblock_x,
        output_dtype=dtype,
    )

    grad_output_dq = convert_from_mxfp8_pytorch(
        grad_output_lp, grad_output_scales, torch.float32, BLOCK_SIZE_DEFAULT, grad_axis, use_2dblock_x)
    inputs_dq = convert_from_mxfp8_pytorch(
        inputs_lp, input_scales, torch.float32, BLOCK_SIZE_DEFAULT, input_axis, use_2dblock_x)
    grad_weights_ref = _reference_wgrad(grad_output_dq, inputs_dq, expert_indices, num_experts, trans_weights)

    cos = _cossim(grad_weights, grad_weights_ref)
    assert cos > 0.999, f"wgrad kernel vs dequant-matmul cos-sim too low: {cos} (2d_x={use_2dblock_x})"


def _run_autograd_case(trans_weights, use_2dblock_x, use_2dblock_w, shape=(256, 128, 128, 2)):
    m_total, n_dim, k_dim, num_experts = shape
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((m_total, k_dim), dtype).requires_grad_(True)
    weight_shape = (num_experts, n_dim, k_dim) if trans_weights else (num_experts, k_dim, n_dim)
    expert_weights = prepare_data(weight_shape, dtype).requires_grad_(True)
    expert_indices = _make_indices(m_total // ALIGN_SIZE_M, num_experts, device)
    target = prepare_data((m_total, n_dim), dtype)

    outputs_ref = _reference(inputs, expert_weights, expert_indices, trans_weights)
    torch.nn.functional.mse_loss(outputs_ref, target).backward()
    grad_inputs_ref = inputs.grad.detach().clone()
    grad_weights_ref = expert_weights.grad.detach().clone()
    inputs.grad.zero_()
    expert_weights.grad.zero_()

    outputs = mxfp8_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )
    torch.nn.functional.mse_loss(outputs, target).backward()

    assert _cossim(outputs, outputs_ref) > 0.99
    assert _cossim(inputs.grad, grad_inputs_ref) > 0.99
    assert _cossim(expert_weights.grad, grad_weights_ref) > 0.99


@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
def test_mxfp8_grouped_gemm_autograd(use_2dblock_x, use_2dblock_w):
    _run_autograd_case(
        trans_weights=True,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
    )


def test_mxfp8_grouped_gemm_autograd_trans_weights_false():
    _run_autograd_case(
        trans_weights=False,
        use_2dblock_x=False,
        use_2dblock_w=False,
        shape=(256, 256, 128, 2),
    )
