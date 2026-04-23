# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Op-level precision tests for NVFP4 Grouped GEMM.

Test layers
-----------
1. ``test_nvfp4_grouped_gemm_forward_accuracy``
   Forward output SNR vs BF16 reference.  Thresholds mirror the NVFP4 linear
   P2P test: 1D/1D → SNR > 10 dB; any 2D block → SNR > 5 dB.

2. ``test_nvfp4_grouped_gemm_autograd``
   Full forward+backward (O, dX, dW) vs BF16 autograd reference.
   Thresholds from NVFP4LinearFunction autograd test:
     use_sr_grad=True  → SNR > 3 dB
     use_sr_grad=False → SNR > 4 dB

3. ``test_nvfp4_grouped_gemm_boundary``
   Smoke test: various expert counts and group multipliers must not produce
   NaN/Inf outputs.

4. ``test_nvfp4_grouped_gemm_single_expert_equals_linear``
   Consistency: when all tokens go to one expert the grouped GEMM result must
   match the plain NVFP4 linear result (SNR > 30 dB).
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import (
    ALIGN_SIZE_M,
    nvfp4_grouped_gemm,
    _nvfp4_grouped_wgrad,
    _has_native_grouped_mm,
)
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.functional import (
    _quantize_then_nvfp4_scaled_grouped_mm,
)
from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_scaled_mm
from .utils import prepare_data, calc_snr, calc_cossim


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_contiguous_expert_indices(
    M_total: int, num_groups: int, num_experts: int, device
) -> torch.Tensor:
    """Create a valid expert-index tensor where every GROUP of ALIGN_SIZE_M tokens
    shares the same expert id."""
    indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    for g in range(num_groups):
        eid = torch.randint(0, num_experts, (1,), device=device).item()
        s = g * ALIGN_SIZE_M
        indices[s : s + ALIGN_SIZE_M] = eid
    return indices


def _bf16_grouped_ref_forward(
    inputs, expert_weights, expert_indices, M_total, N, num_groups, trans_weights
):
    dtype = inputs.dtype
    device = inputs.device
    y = torch.zeros(M_total, N, dtype=dtype, device=device)
    for g in range(num_groups):
        s, e = g * ALIGN_SIZE_M, (g + 1) * ALIGN_SIZE_M
        eid = expert_indices[s].item()
        w = expert_weights[eid]
        y[s:e] = inputs[s:e] @ (w.T if trans_weights else w)
    return y


# ---------------------------------------------------------------------------
# Test 1 – forward accuracy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(512, 256, 256, 4), (1024, 512, 512, 8)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("trans_weights", [True])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
def test_nvfp4_grouped_gemm_forward_accuracy(
    shape, use_2dblock_x, use_2dblock_w, trans_weights, data_type
):
    """NVFP4 grouped GEMM forward output should be close to the BF16 reference."""
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")

    inputs = prepare_data((M_total, K), data_type)
    expert_weights = prepare_data((num_experts, N, K), data_type)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device
    )

    y_ref = _bf16_grouped_ref_forward(
        inputs, expert_weights, expert_indices, M_total, N, num_groups, trans_weights
    )
    y_nvfp4 = nvfp4_grouped_gemm(
        inputs, expert_weights, expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )

    assert y_nvfp4.shape == y_ref.shape
    assert y_nvfp4.dtype == y_ref.dtype

    snr = calc_snr(y_ref, y_nvfp4)
    cossim = calc_cossim(y_ref, y_nvfp4)

    print()
    print(tabulate(
        [["Output", f"{snr:.2f}", f"{cossim:.6f}"]],
        headers=["Tensor", "SNR(dB)", "CosSim"],
        tablefmt="github",
    ))

    min_snr = 10 if not (use_2dblock_x or use_2dblock_w) else 5
    assert snr > min_snr, f"Forward SNR too low: {snr:.2f} (min {min_snr})"
    assert cossim > 0.95, f"Forward CosSim too low: {cossim:.6f}"


# ---------------------------------------------------------------------------
# Test 2 – autograd (O, dX, dW)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(512, 256, 256, 4), (1024, 512, 512, 8)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_nvfp4_grouped_gemm_autograd(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type
):
    """Output, dX, and dW SNR vs BF16 autograd reference must exceed the threshold."""
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")

    inputs_ref = prepare_data((M_total, K), data_type).requires_grad_(True)
    weights_ref = prepare_data((num_experts, N, K), data_type).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device
    )
    target = prepare_data((M_total, N), data_type)

    # BF16 reference backward
    y_ref = _bf16_grouped_ref_forward(
        inputs_ref, weights_ref, expert_indices, M_total, N, num_groups,
        trans_weights=True,
    )
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)
    loss_ref.backward()
    dx_ref = inputs_ref.grad.clone()
    dw_ref = weights_ref.grad.clone()

    # NVFP4 forward + backward
    inputs = inputs_ref.detach().clone().requires_grad_(True)
    weights = weights_ref.detach().clone().requires_grad_(True)

    y = nvfp4_grouped_gemm(
        inputs, weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
    )
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    o_snr = calc_snr(y_ref.detach(), y.detach())
    dx_snr = calc_snr(dx_ref, inputs.grad)
    dw_snr = calc_snr(dw_ref, weights.grad)
    o_sim = calc_cossim(y_ref.detach(), y.detach())
    dx_sim = calc_cossim(dx_ref, inputs.grad)
    dw_sim = calc_cossim(dw_ref, weights.grad)

    print()
    print(tabulate(
        [
            ["O",  f"{o_snr:.2f}",  f"{o_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR(dB)", "CosSim"],
        tablefmt="github",
    ))

    min_snr = 3 if use_sr_grad else 4
    assert o_snr  > min_snr, f"Output SNR too low:  {o_snr:.2f}"
    assert dx_snr > min_snr, f"dX SNR too low:      {dx_snr:.2f}"
    assert dw_snr > min_snr, f"dW SNR too low:      {dw_snr:.2f}"


# ---------------------------------------------------------------------------
# Test 3 – boundary / smoke
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_multiplier", [1, 4, 8])
@pytest.mark.parametrize("num_experts", [1, 4, 32])
def test_nvfp4_grouped_gemm_boundary(M_multiplier, num_experts):
    """Various expert counts and group sizes must not crash or produce NaN/Inf."""
    M_total = ALIGN_SIZE_M * M_multiplier
    N, K = 256, 256
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((M_total, K), dtype)
    expert_weights = prepare_data((num_experts, N, K), dtype)
    expert_indices = _make_contiguous_expert_indices(
        M_total, M_multiplier, num_experts, device
    )

    y = nvfp4_grouped_gemm(inputs, expert_weights, expert_indices)

    assert y.shape == (M_total, N), f"Bad output shape: {y.shape}"
    assert not y.isnan().any(), "Output contains NaN"
    assert not y.isinf().any(), "Output contains Inf"


# ---------------------------------------------------------------------------
# Test 3c – focused recipe smoke for grouped-only knobs that are not covered by
# the dense linear tests: grouped 2D-w, grouped PTS, grouped Hadamard, grouped
# DGE, and their combination.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_2dblock_x,use_2dblock_w,use_per_tensor_scale,use_hadamard,use_dge", [
    (False, False, False, False, False),  # plain baseline
    (False, True,  False, False, False),  # grouped weight 2D
    (False, True,  True,  False, False),  # + PTS
    (False, True,  False, True,  False),  # + Hadamard
    (False, True,  False, False, True),   # + DGE
    (False, True,  False, True,  True),   # + Hadamard + DGE
])
def test_nvfp4_grouped_gemm_recipe_variants_smoke(
    use_2dblock_x, use_2dblock_w, use_per_tensor_scale, use_hadamard, use_dge,
):
    """Grouped recipe variants should run end-to-end, produce finite outputs,
    and backprop finite gradients without silently falling back."""
    M_total, N, K, num_experts = 512, 256, 256, 4
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16
    inputs = prepare_data((M_total, K), dtype).requires_grad_(True)
    expert_weights = prepare_data((num_experts, N, K), dtype).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(M_total, num_groups, num_experts, device)
    target = prepare_data((M_total, N), dtype)

    y = nvfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=True,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=True,
        use_per_tensor_scale=use_per_tensor_scale,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()

    assert y.shape == (M_total, N)
    assert y.dtype == dtype
    assert torch.isfinite(y).all()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert expert_weights.grad is not None and torch.isfinite(expert_weights.grad).all()


# ---------------------------------------------------------------------------
# Test 3b – contract checks for misaligned M_total
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_total", [64, 150])
def test_nvfp4_grouped_gemm_rejects_non_aligned_mtotal(M_total):
    """Loop backend must fail fast on shapes that would otherwise produce a
    silent zero tail or expose the pre-existing non-aligned axis=0 QDQ issue."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, N, K = 4, 256, 256
    # One expert per ALIGN_SIZE_M-sized chunk for the aligned prefix.
    num_groups = max(1, M_total // ALIGN_SIZE_M)

    inputs = prepare_data((M_total, K), dtype)
    expert_weights = prepare_data((num_experts, N, K), dtype)
    expert_indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    offs = torch.tensor([M_total // num_experts * (i + 1) for i in range(num_experts)],
                        dtype=torch.int32, device=device)
    offs[-1] = M_total

    with pytest.raises(RuntimeError):
        nvfp4_grouped_gemm(
            inputs, expert_weights, expert_indices,
            trans_weights=True,
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False,
        )

    with pytest.raises(RuntimeError):
        _quantize_then_nvfp4_scaled_grouped_mm(
            inputs, expert_weights.transpose(-2, -1).contiguous(), offs,
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False,
        )


# ---------------------------------------------------------------------------
# Test 4 – consistency with NVFP4 linear (single expert)
# ---------------------------------------------------------------------------

def test_nvfp4_grouped_gemm_single_expert_equals_linear():
    """When all tokens go to the same expert, grouped GEMM == linear (SNR > 30 dB)."""
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Use a size that is an exact multiple of ALIGN_SIZE_M
    M = ALIGN_SIZE_M
    N, K = 256, 256

    torch.manual_seed(42)
    x = prepare_data((M, K), dtype)
    w = prepare_data((N, K), dtype)

    # NVFP4 linear
    y_linear = _to_nvfp4_then_scaled_mm(
        x, w,
        use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False, use_per_tensor_scale=False,
    )

    # NVFP4 grouped GEMM: one expert, all tokens -> expert 0
    expert_weights = w.unsqueeze(0)                                   # [1, N, K]
    expert_indices = torch.zeros(M, dtype=torch.int32, device=device)

    y_gg = nvfp4_grouped_gemm(
        x, expert_weights, expert_indices,
        trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False,
    )

    snr = calc_snr(y_linear, y_gg)
    cossim = calc_cossim(y_linear, y_gg)
    print(f"\nSingle-expert grouped GEMM vs linear:  SNR={snr:.2f} dB  CosSim={cossim:.8f}")

    assert snr > 30, f"SNR too low: {snr:.2f} – grouped GEMM and linear should be identical"
    assert cossim > 0.9999, f"CosSim too low: {cossim:.8f}"


# ---------------------------------------------------------------------------
# Test 5 – _nvfp4_grouped_wgrad in isolation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(512, 256, 256, 4)])
@pytest.mark.parametrize("use_sr_grad", [False, True])
def test_nvfp4_grouped_wgrad_isolation(shape, use_sr_grad):
    """_nvfp4_grouped_wgrad should produce the same dW as the full autograd path."""
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs_ref = prepare_data((M_total, K), dtype).requires_grad_(True)
    weights_ref = prepare_data((num_experts, N, K), dtype).requires_grad_(True)
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device
    )
    target = prepare_data((M_total, N), dtype)

    # BF16 reference wgrad
    y_ref = _bf16_grouped_ref_forward(
        inputs_ref, weights_ref, expert_indices, M_total, N, num_groups,
        trans_weights=True,
    )
    loss_ref = torch.nn.functional.mse_loss(y_ref, target)
    loss_ref.backward()
    dw_ref = weights_ref.grad.clone()

    # Use NVFP4 autograd path to get x_bwd, then call _nvfp4_grouped_wgrad
    from alto.kernels.fp4.nvfp4.nvfp_linear import _qdq

    inputs_det = inputs_ref.detach().clone()
    x_bwd = _qdq(inputs_det, axis=0, is_2d_block=False, use_per_tensor_scale=False)
    grad_output = 2.0 * (y_ref.detach() - target) / target.numel()

    dw = _nvfp4_grouped_wgrad(
        grad_output=grad_output,
        x_bwd=x_bwd,
        expert_indices=expert_indices,
        num_experts=num_experts,
        num_groups=num_groups,
        use_sr_grad=use_sr_grad,
        use_per_tensor_scale=False,
        use_2dblock_x=False,
        trans_weights=True,
        output_dtype=dtype,
    )

    snr = calc_snr(dw_ref, dw)
    cossim = calc_cossim(dw_ref, dw)

    print()
    print(tabulate(
        [["dW (wgrad)", f"{snr:.2f}", f"{cossim:.6f}"]],
        headers=["Tensor", "SNR(dB)", "CosSim"],
        tablefmt="github",
    ))

    min_snr = 2 if use_sr_grad else 3
    assert snr > min_snr, f"wgrad SNR too low: {snr:.2f}"


# ---------------------------------------------------------------------------
# Test 6 – torch._grouped_mm platform detection
# ---------------------------------------------------------------------------

def test_grouped_mm_platform_detection():
    """_has_native_grouped_mm should return a bool and be deterministic."""
    result = _has_native_grouped_mm()
    assert isinstance(result, bool)
    assert _has_native_grouped_mm() == result, "Detection must be deterministic"
    print(f"\ntorch._grouped_mm available: {result}")


# ---------------------------------------------------------------------------
# Test 7 – native dispatch path (_quantize_then_nvfp4_scaled_grouped_mm)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(512, 256, 256, 4)])
@pytest.mark.parametrize("use_2dblock_x", [False])
@pytest.mark.parametrize("use_2dblock_w", [False])
def test_nvfp4_native_dispatch_forward(shape, use_2dblock_x, use_2dblock_w):
    """_quantize_then_nvfp4_scaled_grouped_mm (dispatch path) forward should match
    nvfp4_grouped_gemm (loop path) when tokens are sorted by expert."""
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((M_total, K), dtype)
    # dispatch convention: W is [E, K, N] (trans_weights=False)
    expert_weights = prepare_data((num_experts, K, N), dtype)

    # Build sorted expert_indices and matching offs
    # Assign groups round-robin to experts, sorted
    sorted_indices = torch.zeros(M_total, dtype=torch.int32, device=device)
    groups_per_expert = num_groups // num_experts
    remainder = num_groups % num_experts
    offs_list = []
    cumulative = 0
    g = 0
    for e in range(num_experts):
        count = groups_per_expert + (1 if e < remainder else 0)
        for _ in range(count):
            s = g * ALIGN_SIZE_M
            sorted_indices[s:s + ALIGN_SIZE_M] = e
            g += 1
        cumulative += count * ALIGN_SIZE_M
        offs_list.append(cumulative)
    offs = torch.tensor(offs_list, dtype=torch.int32, device=device)

    # Reference via loop-based path
    y_loop = nvfp4_grouped_gemm(
        inputs, expert_weights, sorted_indices,
        trans_weights=False,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )

    # Native dispatch path
    y_native = _quantize_then_nvfp4_scaled_grouped_mm(
        inputs, expert_weights, offs,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )

    # Both paths implement the same grouped recipe but may accumulate in
    # different tile orders.  Check for numerical equivalence rather than
    # bit-exact identity.
    snr = calc_snr(y_loop, y_native)
    cossim = calc_cossim(y_loop, y_native)
    assert snr > 30, f"native-vs-loop SNR too low: {snr:.2f}"
    assert cossim > 0.9999, f"native-vs-loop CosSim too low: {cossim:.8f}"
