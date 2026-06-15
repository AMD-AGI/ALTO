# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_forward import mxfp8_grouped_gemm_forward
from alto.kernels.fp4.testing_utils import calc_snr

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
@pytest.mark.parametrize("trans_weights", [True, False])
def test_forward(shape, use_2dblock_x, use_2dblock_w, trans_weights):
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    data_type = torch.bfloat16

    inputs = prepare_data((M_total, K), data_type)
    weight_shape = (num_experts, N, K) if trans_weights else (num_experts, K, N)
    weight_axis = -1 if trans_weights else -2
    expert_weights = prepare_data(weight_shape, data_type)
    indices = _make_indices(num_groups, num_experts, device)

    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=use_2dblock_x)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=weight_axis, is_2d_block=use_2dblock_w)

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
    w_dq = convert_from_mxfp8_pytorch(w_lp, w_s, torch.float32, BLOCK_SIZE_DEFAULT, weight_axis, use_2dblock_w)
    ref_dq = _reference(x_dq, w_dq, indices, num_groups, trans_weights)
    cos_dq = _cossim(out, ref_dq)
    assert cos_dq > 0.999, \
        f"kernel vs dequant-matmul cos-sim too low: {cos_dq} (shape={shape}, 2dx={use_2dblock_x}, 2dw={use_2dblock_w})"
    # SNR catches magnitude errors (missed scale, wrong accumulator) that cossim is blind to.
    snr_dq = calc_snr(ref_dq, out.float())
    assert snr_dq > 40, \
        f"kernel vs dequant-matmul SNR too low: {snr_dq:.1f}dB (shape={shape}, 2dx={use_2dblock_x}, 2dw={use_2dblock_w})"

    # Looser sanity check against the full-precision bf16 path.
    ref_bf16 = _reference(inputs, expert_weights, indices, num_groups, trans_weights)
    cos_bf16 = _cossim(out, ref_bf16)
    assert cos_bf16 > 0.99, f"kernel vs bf16 cos-sim too low: {cos_bf16}"


def test_forward_dequant_fallback_matches_dot_scaled():
    """CDNA3 path (use_dot_scaled=False) must match the dequant-matmul reference.

    Forced on regardless of the running device so the _dequantize_fp8 -> tl.dot
    branch is covered; real MI300 ground-truth still needs Step 7 on CDNA3.
    """
    M_total, N, K, num_experts = 256, 128, 128, 2
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    data_type = torch.bfloat16

    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((num_experts, N, K), data_type)
    indices = _make_indices(num_groups, num_experts, device)

    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=True)

    out = mxfp8_grouped_gemm_forward(
        x_lp, w_lp, indices, x_s, w_s,
        trans_weights=True,
        use_2dblock_x=False,
        use_2dblock_w=True,
        output_dtype=data_type,
        use_dot_scaled=False,
    )

    x_dq = convert_from_mxfp8_pytorch(x_lp, x_s, torch.float32, BLOCK_SIZE_DEFAULT, -1, False)
    w_dq = convert_from_mxfp8_pytorch(w_lp, w_s, torch.float32, BLOCK_SIZE_DEFAULT, -1, True)
    ref_dq = _reference(x_dq, w_dq, indices, num_groups, trans_weights=True)
    cos = _cossim(out, ref_dq)
    assert cos > 0.999, f"fallback forward vs dequant-matmul cos-sim too low: {cos}"


def test_forward_single_expert_matches_mxfp8_linear():
    """All tokens to one expert => grouped GEMM must match mxfp8 linear (x @ w^T).

    Independent cross-check: the linear path quantizes via its own autograd
    Function, so this catches bugs the dequant-matmul reference can't — that
    reference shares this test's convert_to_mxfp8 output, so a quant bug would
    corrupt both sides equally and still pass.
    """
    from alto.kernels.mxfp8.mxfp8_linear import _to_mxfp8_then_scaled_mm

    M, N, K = ALIGN_SIZE_M, 256, 256
    data_type = torch.bfloat16
    x = prepare_data((M, K), data_type)
    w = prepare_data((N, K), data_type)
    indices = torch.zeros(M, dtype=torch.int32, device="cuda")

    y_linear = _to_mxfp8_then_scaled_mm(x, w, use_2dblock_x=False, use_2dblock_w=False)

    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        x, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        w.unsqueeze(0), block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    y_gg = mxfp8_grouped_gemm_forward(
        x_lp, w_lp, indices, x_s, w_s, trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False, output_dtype=data_type)

    snr = calc_snr(y_linear.float(), y_gg.float())
    assert snr > 30, f"single-expert grouped GEMM vs mxfp8 linear SNR too low: {snr:.1f}dB"


def test_forward_rejects_indices_length_mismatch():
    M_total, N, K, num_experts = ALIGN_SIZE_M, 128, 128, 1
    data_type = torch.bfloat16
    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((num_experts, N, K), data_type)
    # A routed count below the buffer is now the padded-buffer case, not an error;
    # but a non-128-aligned routed count (127) must still be rejected.
    indices = torch.zeros(M_total - 1, dtype=torch.int32, device="cuda")
    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)

    with pytest.raises(AssertionError, match="multiple of group_size_m"):
        mxfp8_grouped_gemm_forward(x_lp, w_lp, indices, x_s, w_s)


def test_forward_rejects_non_aligned_mtotal():
    M_total, N, K = 64, 128, 128  # 64 not a multiple of ALIGN_SIZE_M (128)
    data_type = torch.bfloat16
    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((1, N, K), data_type)
    indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)

    with pytest.raises(AssertionError, match="multiple of group_size_m"):
        mxfp8_grouped_gemm_forward(x_lp, w_lp, indices, x_s, w_s)


def test_forward_rejects_wrong_scale_shape():
    M_total, N, K, num_experts = ALIGN_SIZE_M, 128, 128, 1
    data_type = torch.bfloat16
    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((num_experts, N, K), data_type)
    indices = torch.zeros(M_total, dtype=torch.int32, device="cuda")
    x_lp, x_s = torch.ops.alto.convert_to_mxfp8(
        inputs, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)
    w_lp, w_s = torch.ops.alto.convert_to_mxfp8(
        expert_weights, block_size=BLOCK_SIZE_DEFAULT, mxfp_format="e4m3", axis=-1, is_2d_block=False)

    with pytest.raises(AssertionError, match="weight_scales shape"):
        mxfp8_grouped_gemm_forward(x_lp, w_lp, indices, x_s, w_s[:, :, :-1])
