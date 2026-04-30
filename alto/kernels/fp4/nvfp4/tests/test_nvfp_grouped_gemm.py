# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Op-level precision tests for NVFP4 Grouped GEMM.

Test layers
-----------
1. ``test_nvfp4_grouped_gemm_autograd``
   Full forward+backward (O, dX, dW) vs BF16 autograd reference.
   Uses the shared K-aware NVFP4 autograd SNR helper so ordinary cases keep a
   tight floor while large-K SR stress cases remain valid.
   Also enforces the tighter forward SNR bound on the non-SR BF16 path so
   regressions in the QDQ / fprop kernels get caught early.

2. ``test_nvfp4_grouped_gemm_boundary``
   Smoke test on shape corner cases (single group / 1-expert / 32-expert).

3. ``test_nvfp4_grouped_gemm_recipe_variants_smoke``
   Recipe knobs (2D-w, PTS, Hadamard, DGE, combinations) run end-to-end and
   produce finite outputs + gradients.

4. ``test_nvfp4_grouped_gemm_rejects_non_aligned_mtotal``
   Contract guard: loop backend must fail fast on misaligned M_total.

5. ``test_nvfp4_grouped_gemm_single_expert_equals_linear``
   Consistency: when all tokens go to one expert the grouped GEMM result must
   match the plain NVFP4 linear result (SNR > 30 dB).

6. ``test_nvfp4_native_dispatch_forward``
   Dispatch path (_quantize_then_nvfp4_scaled_grouped_mm, offs-based) must
   numerically match the user-facing path (nvfp4_grouped_gemm, indices-based).
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm import mxfp4_grouped_gemm
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import (
    ALIGN_SIZE_M,
    nvfp4_grouped_gemm,
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
# Test 1 – autograd (O, dX, dW)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [
    (512, 256, 256, 4),
    (1024, 512, 512, 8),
    (512, 512, 2048, 4),
])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_nvfp4_grouped_gemm_autograd(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type
):
    """Output, dX, and dW SNR vs BF16 autograd reference must remain healthy."""
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

    check_nvfp4_autograd_snr(
        {"O": o_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=use_sr_grad,
        kind="nvfp4_grouped_gemm",
        context=(
            f"NVFP4GroupedGEMM shape={shape} dtype={data_type} "
            f"x_2d={use_2dblock_x} w_2d={use_2dblock_w}"
        ),
    )

    # Tight forward-path floor on the deterministic (non-SR) BF16 path.
    # This replaces the previous ``test_nvfp4_grouped_gemm_forward_accuracy``:
    # the old 1D/1D → 10 dB and any-2D → 5 dB bounds are preserved here so
    # regressions in QDQ / fprop kernels still get caught, without paying for
    # a second parametrized test with overlapping coverage.
    if not use_sr_grad and data_type == torch.bfloat16:
        fwd_min_snr = 10 if not (use_2dblock_x or use_2dblock_w) else 5
        assert o_snr > fwd_min_snr, (
            f"Forward SNR too low on non-SR BF16 path: {o_snr:.2f} "
            f"(min {fwd_min_snr})"
        )
        assert o_sim > 0.95, f"Forward CosSim too low: {o_sim:.6f}"


@pytest.mark.parametrize("shape,trans_weights,use_2dblock_x,use_2dblock_w", [
    # Shared with the MXFP4 grouped-GEMM test matrix.  Cover both expert-weight
    # layouts without adding the full MXFP4 combinatorial grid here.
    ((1024, 512, 512, 8), True,  False, False),
    ((1024, 512, 512, 8), False, False, True),
])
def test_nvfp4_grouped_gemm_forward_compares_with_mxfp4(
    shape, trans_weights, use_2dblock_x, use_2dblock_w,
):
    """NVFP4 and MXFP4 grouped forward paths should both track BF16 reference.

    The formats are not expected to match each other bitwise.  This comparison
    only checks that both grouped-GEMM families run on representative
    routed-expert cases that already exist in both original test suites and
    remain within a reasonable forward-error band against BF16.
    """
    M_total, N, K, num_experts = shape
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((M_total, K), dtype)
    expert_weights = prepare_data(
        (num_experts, N, K) if trans_weights else (num_experts, K, N),
        dtype,
    )
    expert_indices = _make_contiguous_expert_indices(
        M_total, num_groups, num_experts, device,
    )
    # Build one BF16 routed-expert reference and compare both FP4 formats
    # against it independently.  The test does not assert NVFP4 and MXFP4 are
    # numerically identical; their scale formats and quantization errors differ.
    y_ref = _bf16_grouped_ref_forward(
        inputs,
        expert_weights,
        expert_indices,
        M_total,
        N,
        num_groups,
        trans_weights=trans_weights,
    )

    # Use analogous minimal grouped recipes for both format families.  In
    # particular, do not enable MXFP4 clipping or macro-block scaling here:
    # those are valuable recipe knobs, but this test is only a cross-format
    # smoke/reference check for the common routed-expert shape.
    y_nv = nvfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
    )
    y_mx = mxfp4_grouped_gemm(
        inputs,
        expert_weights,
        expert_indices,
        trans_weights=trans_weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
        use_dge=False,
        use_hadamard=False,
        clip_mode="none",
        use_macro_block_scaling=False,
    )

    nv_snr = calc_snr(y_ref, y_nv)
    mx_snr = calc_snr(y_ref, y_mx)
    print()
    print(tabulate(
        [
            ["NVFP4", f"{nv_snr:.2f}", f"{calc_cossim(y_ref, y_nv):.6f}"],
            ["MXFP4", f"{mx_snr:.2f}", f"{calc_cossim(y_ref, y_mx):.6f}"],
        ],
        headers=["Format", "SNR", "CosSim"], tablefmt="github",
    ))

    assert torch.isfinite(y_nv).all()
    assert torch.isfinite(y_mx).all()
    # Forward SNR against BF16 is the invariant we care about.  A 10 dB floor
    # catches wrong routing / scale handling while avoiding brittle assumptions
    # about one FP4 format matching the other.
    assert nv_snr > 10, f"NVFP4 grouped forward SNR too low: {nv_snr:.2f}"
    assert mx_snr > 10, f"MXFP4 grouped forward SNR too low: {mx_snr:.2f}"


# ---------------------------------------------------------------------------
# Test 2 – boundary / smoke (corner cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_multiplier,num_experts", [
    (1, 1),   # smallest possible grouped run
    (1, 32),  # many experts, few tokens
    (8, 1),   # many tokens, single expert
    (8, 32),  # many tokens, many experts
])
def test_nvfp4_grouped_gemm_boundary(M_multiplier, num_experts):
    """Corner-case expert / group counts must not crash or produce NaN/Inf."""
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
# Test 3 – focused recipe smoke for grouped-only knobs that are not covered by
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
# Test 4 – contract checks for misaligned M_total
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M_total", [64, 150])
def test_nvfp4_grouped_gemm_rejects_non_aligned_mtotal(M_total):
    """Loop backend must fail fast on shapes that would otherwise produce a
    silent zero tail or expose the pre-existing non-aligned axis=0 QDQ issue."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, N, K = 4, 256, 256

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
# Test 5 – consistency with NVFP4 linear (single expert)
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
# Test 6 – native dispatch path (_quantize_then_nvfp4_scaled_grouped_mm)
# ---------------------------------------------------------------------------

def test_nvfp4_native_dispatch_forward():
    """_quantize_then_nvfp4_scaled_grouped_mm (dispatch path) forward should match
    nvfp4_grouped_gemm (loop path) when tokens are sorted by expert."""
    M_total, N, K, num_experts = 512, 256, 256, 4
    M_total = (M_total // ALIGN_SIZE_M) * ALIGN_SIZE_M
    num_groups = M_total // ALIGN_SIZE_M
    device = torch.device("cuda")
    dtype = torch.bfloat16

    inputs = prepare_data((M_total, K), dtype)
    # dispatch convention: W is [E, K, N] (trans_weights=False)
    expert_weights = prepare_data((num_experts, K, N), dtype)

    # Build sorted expert_indices and matching offs.  Assign groups
    # round-robin to experts, sorted.
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
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_sr_grad=False,
    )

    # Native dispatch path
    y_native = _quantize_then_nvfp4_scaled_grouped_mm(
        inputs, expert_weights, offs,
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_sr_grad=False,
    )

    # Both paths implement the same grouped recipe but may accumulate in
    # different tile orders.  Check for numerical equivalence rather than
    # bit-exact identity.
    snr = calc_snr(y_loop, y_native)
    cossim = calc_cossim(y_loop, y_native)
    assert snr > 30, f"native-vs-loop SNR too low: {snr:.2f}"
    assert cossim > 0.9999, f"native-vs-loop CosSim too low: {cossim:.8f}"
