# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Op-level precision tests for MXFP8 contiguous grouped GEMM.

Forward (kernel-level correctness), full autograd (O/dX/dW), and the offsets
dispatch entry all live here, mirroring the single-file mxfp4/nvfp4 grouped-GEMM
test layout.
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.mxfp8.mxfp8_quantization import BLOCK_SIZE_DEFAULT
from alto.kernels.mxfp8.mxfp8_grouped_gemm import (
    mxfp8_grouped_gemm,
    _quantize_then_mxfp8_scaled_grouped_mm,
)
from alto.kernels.mxfp8.mxfp8_grouped_gemm.autotune import ALIGN_SIZE_M
from alto.kernels.mxfp8.mxfp8_grouped_gemm.cg_forward import mxfp8_grouped_gemm_forward

from .utils import prepare_data, convert_from_mxfp8_pytorch, calc_snr, calc_cossim, make_indices


def _reference(inputs, expert_weights, indices, trans_weights):
    """BF16/fp32 grouped matmul reference: Y[g] = X[g] @ W[expert(g)]^T per group."""
    m_total = inputs.shape[0]
    n_dim = expert_weights.shape[1] if trans_weights else expert_weights.shape[2]
    out = torch.zeros((m_total, n_dim), dtype=inputs.dtype, device=inputs.device)
    for start in range(0, m_total, ALIGN_SIZE_M):
        expert_idx = indices[start].item()
        weight = expert_weights[expert_idx].t() if trans_weights else expert_weights[expert_idx]
        out[start:start + ALIGN_SIZE_M] = inputs[start:start + ALIGN_SIZE_M] @ weight
    return out


# ---------------------------------------------------------------------------
# Forward – kernel-level correctness on pre-quantized operands
# ---------------------------------------------------------------------------

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
    indices = make_indices(num_groups, num_experts, device)

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

    tag = f"shape={shape}, 2dx={use_2dblock_x}, 2dw={use_2dblock_w}"

    # Primary correctness gate: dequantize the exact fp8 operands the kernel
    # consumed, matmul in PyTorch. This isolates kernel-port correctness from
    # mxfp8-vs-bf16 quantization error.
    x_dq = convert_from_mxfp8_pytorch(x_lp, x_s, torch.float32, BLOCK_SIZE_DEFAULT, -1, use_2dblock_x)
    w_dq = convert_from_mxfp8_pytorch(w_lp, w_s, torch.float32, BLOCK_SIZE_DEFAULT, weight_axis, use_2dblock_w)
    ref_dq = _reference(x_dq, w_dq, indices, trans_weights)
    cos_dq = calc_cossim(out, ref_dq)
    assert cos_dq > 0.999, f"kernel vs dequant-matmul cos-sim too low: {cos_dq} ({tag})"
    # SNR catches magnitude errors (missed scale, wrong accumulator) that cossim is blind to.
    snr_dq = calc_snr(ref_dq, out.float())
    assert snr_dq > 40, f"kernel vs dequant-matmul SNR too low: {snr_dq:.1f}dB ({tag})"

    # Looser sanity check against the full-precision bf16 path.
    ref_bf16 = _reference(inputs, expert_weights, indices, trans_weights)
    cos_bf16 = calc_cossim(out, ref_bf16)
    assert cos_bf16 > 0.99, f"kernel vs bf16 cos-sim too low: {cos_bf16} ({tag})"


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


# ---------------------------------------------------------------------------
# Autograd – forward+backward (O, dX, dW) vs a BF16 autograd reference
# ---------------------------------------------------------------------------

# Mirrors the mxfp4/nvfp4 grouped-GEMM autograd tests. The user-facing path
# quantizes internally, so the kernel's dgrad and wgrad are exercised here
# through real backprop (no separate dgrad/wgrad kernel-wrapper tests needed).
#
# Floors are set from the measured min/max across this full 32-case grid (vs the
# BF16 autograd reference, so the spread is e4m3 quantization error, not a bug):
#     O  SNR 24.9 .. 28.8 dB   cossim 0.999 .. 0.999
#     dX SNR 19.0 .. 26.5 dB   cossim 0.996 .. 0.999
#     dW SNR 19.0 .. 27.0 dB   cossim 0.996 .. 0.999
# SNR is the real precision gate (far stricter than fp4's 10/9 dB bounds, with
# ~4-5 dB headroom below the observed minima). cossim is only a coarse
# direction-sanity backstop, kept loose at 0.99 against the 0.996 worst case.
@pytest.mark.parametrize("shape", [
    (384, 128, 128, 2),
    (512, 256, 256, 4),
    (512, 512, 2048, 4),   # large-K stress
    (384, 256, 128, 2),    # N != K, exercises non-square weight layout / strides
])
@pytest.mark.parametrize("trans_weights", [True, False])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
def test_mxfp8_grouped_gemm_autograd(shape, trans_weights, use_2dblock_x, use_2dblock_w):
    m_total, n_dim, k_dim, num_experts = shape
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((m_total, k_dim), dtype).requires_grad_(True)
    weight_shape = (num_experts, n_dim, k_dim) if trans_weights else (num_experts, k_dim, n_dim)
    expert_weights = prepare_data(weight_shape, dtype).requires_grad_(True)
    expert_indices = make_indices(m_total // ALIGN_SIZE_M, num_experts, device)
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

    o_snr = calc_snr(outputs_ref, outputs.detach())
    dx_snr = calc_snr(grad_inputs_ref, inputs.grad)
    dw_snr = calc_snr(grad_weights_ref, expert_weights.grad)
    o_sim = calc_cossim(outputs_ref, outputs.detach())
    dx_sim = calc_cossim(grad_inputs_ref, inputs.grad)
    dw_sim = calc_cossim(grad_weights_ref, expert_weights.grad)

    print()
    print(tabulate(
        [["O", f"{o_snr:.2f}", f"{o_sim:.6f}"],
         ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
         ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"]],
        headers=["Tensor", "SNR(dB)", "CosSim"], tablefmt="github"))

    tag = f"shape={shape} tr={trans_weights} x2={use_2dblock_x} w2={use_2dblock_w}"
    assert o_snr > 20, f"O SNR too low: {o_snr:.2f} ({tag})"
    assert dx_snr > 15, f"dX SNR too low: {dx_snr:.2f} ({tag})"
    assert dw_snr > 15, f"dW SNR too low: {dw_snr:.2f} ({tag})"
    assert o_sim > 0.99 and dx_sim > 0.99 and dw_sim > 0.99, \
        f"cossim too low: O={o_sim:.4f} dX={dx_sim:.4f} dW={dw_sim:.4f} ({tag})"


def test_autograd_many_experts_with_empty_expert():
    """experts > groups => some experts receive zero tokens (dW row must be 0),
    and the wgrad scan-all-groups path is exercised with finite gradients."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    m_total, n_dim, k_dim, num_experts = ALIGN_SIZE_M * 2, 128, 128, 8  # 2 groups, 8 experts

    inputs = prepare_data((m_total, k_dim), dtype).requires_grad_(True)
    expert_weights = prepare_data((num_experts, n_dim, k_dim), dtype).requires_grad_(True)
    # Route both groups to experts 0 and 1; experts 2..7 stay empty.
    expert_indices = torch.zeros(m_total, dtype=torch.int32, device=device)
    expert_indices[ALIGN_SIZE_M:] = 1
    target = prepare_data((m_total, n_dim), dtype)

    outputs = mxfp8_grouped_gemm(inputs, expert_weights, expert_indices, trans_weights=True)
    torch.nn.functional.mse_loss(outputs, target).backward()

    assert torch.isfinite(outputs).all()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(expert_weights.grad).all()
    # Empty experts must get exactly zero weight gradient.
    assert torch.count_nonzero(expert_weights.grad[2:]) == 0, "unused experts must have zero dW"
    assert torch.count_nonzero(expert_weights.grad[:2]) > 0, "routed experts must have nonzero dW"


# ---------------------------------------------------------------------------
# Offsets dispatch entry + padded buffer (PLAN.md §8.1)
# ---------------------------------------------------------------------------

def test_mxfp8_grouped_gemm_accepts_padded_buffer():
    """Padded activation buffer (M_bufferlen > routed M_total) must not disturb
    routed rows, and padding rows must stay zero in both output and gradients.

    Padded-vs-unpadded self-comparison through the offsets dispatch entry (which
    uses trans_weights=False, also boosting that path's coverage). use_sr_grad is
    forced False so quantization is deterministic and torch.equal holds bitwise.
    """
    device = torch.device("cuda")
    dtype = torch.bfloat16
    M_routed, M_pad = 4 * ALIGN_SIZE_M, 2 * ALIGN_SIZE_M  # 512 routed, 256 padding
    M_bufferlen = M_routed + M_pad
    num_experts, K, N = 4, 256, 256

    routed_rows = prepare_data((M_routed, K), dtype)
    weights = prepare_data((num_experts, K, N), dtype)  # dispatch convention [E, K, N]
    # Each expert owns one ALIGN_SIZE_M group of routed tokens -> cumulative offs.
    offs = torch.tensor(
        [(i + 1) * (M_routed // num_experts) for i in range(num_experts)],
        dtype=torch.int32, device=device,
    )

    inputs_pad = torch.zeros(M_bufferlen, K, dtype=dtype, device=device)
    inputs_pad[:M_routed] = routed_rows
    inputs_pad.requires_grad_(True)
    weights_pad = weights.clone().requires_grad_(True)
    y_pad = _quantize_then_mxfp8_scaled_grouped_mm(
        inputs_pad, weights_pad, offs,
        use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=False)

    inputs_ref = routed_rows.clone().requires_grad_(True)
    weights_ref = weights.clone().requires_grad_(True)
    y_ref = _quantize_then_mxfp8_scaled_grouped_mm(
        inputs_ref, weights_ref, offs,
        use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=False)

    assert y_pad.shape == (M_bufferlen, N)
    assert torch.equal(y_pad[M_routed:], torch.zeros_like(y_pad[M_routed:])), \
        "padding output rows must be zero"
    assert torch.equal(y_pad[:M_routed], y_ref), \
        "routed output rows must be unaffected by buffer padding"

    y_pad.sum().backward()
    y_ref.sum().backward()
    assert inputs_pad.grad.shape == (M_bufferlen, K)
    assert torch.equal(inputs_pad.grad[M_routed:], torch.zeros_like(inputs_pad.grad[M_routed:])), \
        "padding rows must receive zero input gradient"
    assert torch.equal(inputs_pad.grad[:M_routed], inputs_ref.grad), \
        "routed-row input gradients must be unaffected by buffer padding"
    assert torch.equal(weights_pad.grad, weights_ref.grad), \
        "weight gradients must be unaffected by buffer padding"


def test_mxfp8_dispatch_entry_offsets_matches_indices():
    """The offsets entry must equal the indices path: create_indices_from_offsets
    round-trip is correct. No padding here (M_bufferlen == M_total)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_groups, num_experts, K, N = 4, 4, 256, 256
    M_total = num_groups * ALIGN_SIZE_M

    inputs = prepare_data((M_total, K), dtype)
    weights = prepare_data((num_experts, K, N), dtype)  # [E, K, N] dispatch convention
    # Sorted one-group-per-expert routing (group g -> expert g) so it matches the
    # cumulative offs below; the random make_indices helper would not line up.
    indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        indices[g * ALIGN_SIZE_M:(g + 1) * ALIGN_SIZE_M] = g
    offs = torch.arange(1, num_groups + 1, dtype=torch.int32, device=device) * ALIGN_SIZE_M

    y_indices = mxfp8_grouped_gemm(
        inputs, weights, indices, trans_weights=False, use_sr_grad=False)
    y_offsets = _quantize_then_mxfp8_scaled_grouped_mm(
        inputs, weights, offs, use_sr_grad=False)

    snr = calc_snr(y_indices.float(), y_offsets.float())
    assert snr > 30, f"offsets entry vs indices path SNR too low: {snr:.1f}dB"
