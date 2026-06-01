# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT, is_cdna4
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_forward import mxfp8_grouped_gemm_forward

from .utils import prepare_data, convert_from_mxfp8_pytorch


def _cossim(x, y):
    x = x.flatten().to(torch.float32)
    y = y.flatten().to(torch.float32)
    return torch.nn.functional.cosine_similarity(x, y, dim=0).item()


def _make_indices(num_groups, num_experts, device):
    indices = torch.zeros(num_groups * ALIGN_SIZE_M, dtype=torch.int32, device=device)
    for g in range(num_groups):
        e = torch.randint(0, num_experts, (1,), device=device, dtype=torch.int32).item()
        indices[g * ALIGN_SIZE_M:(g + 1) * ALIGN_SIZE_M] = e
    return indices


def _reference(inputs, expert_weights, indices, num_groups, trans_weights):
    M_total, N = inputs.shape[0], (expert_weights.shape[1] if trans_weights else expert_weights.shape[2])
    out = torch.zeros((M_total, N), dtype=inputs.dtype, device=inputs.device)
    for g in range(num_groups):
        s, e = g * ALIGN_SIZE_M, (g + 1) * ALIGN_SIZE_M
        w = expert_weights[indices[s].item()]
        w = w.t() if trans_weights else w
        out[s:e] = inputs[s:e] @ w
    return out


@pytest.mark.parametrize("shape", [(256, 128, 128, 1), (512, 256, 256, 4)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
def test_forward(shape, use_2dblock_x, use_2dblock_w):
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    data_type = torch.bfloat16
    trans_weights = True

    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((num_experts, N, K), data_type)
    indices = _make_indices(num_groups, num_experts, device)

    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=use_2dblock_x)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=use_2dblock_w)

    out = mxfp8_grouped_gemm_forward(
        x_lp, w_lp, indices, x_s, w_s,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        lhs_format_id=0,
        rhs_format_id=0,
        output_dtype=data_type,
    )

    # Primary correctness gate: dequantize the exact fp8 operands the kernel
    # consumed, matmul in PyTorch. This isolates kernel-port correctness from
    # mxfp8-vs-bf16 quantization error.
    x_dq = convert_from_mxfp8_pytorch(x_lp, x_s, torch.float32, BLOCK_SIZE_DEFAULT, -1, use_2dblock_x)
    w_dq = convert_from_mxfp8_pytorch(w_lp, w_s, torch.float32, BLOCK_SIZE_DEFAULT, -1, use_2dblock_w)
    ref_dq = _reference(x_dq, w_dq, indices, num_groups, trans_weights)
    cos_dq = _cossim(out, ref_dq)
    assert cos_dq > 0.999, \
        f"kernel vs dequant-matmul cos-sim too low: {cos_dq} (shape={shape}, 2dx={use_2dblock_x}, 2dw={use_2dblock_w})"

    # Looser sanity check against the full-precision bf16 path.
    ref_bf16 = _reference(inputs, expert_weights, indices, num_groups, trans_weights)
    cos_bf16 = _cossim(out, ref_bf16)
    assert cos_bf16 > 0.99, f"kernel vs bf16 cos-sim too low: {cos_bf16}"
