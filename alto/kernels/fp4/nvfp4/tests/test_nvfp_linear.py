# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Unit tests for the NVFP4 linear op.

Two tests, scoped like ``test_mxfp_linear.py`` but adapted to NVFP4's
software-simulated path (there is no native NVFP4 GEMM kernel yet on AMD, so
there is nothing analogous to ``test_mxfp4_linear_kernel`` to add):

* ``test_nvfp4_qdq_roundtrip`` — per-operand QDQ accuracy, the building
  block that ``NVFP4LinearFunction`` stacks together.  Guards the
  single-pass quantise → dequantise behaviour independently of any GEMM.

* ``test_nvfp4_linear_autograd_function`` — MXFP4-parity check, asserting
  the forward output and both gradients (``dX``, ``dW``) track the BF16
  reference within an SNR threshold.

Recipe knobs such as ``use_hadamard`` / ``use_dge`` are not parametrised here
to keep the matrix small; their correctness under the production dispatch
layer is covered by the parametrised ``test_linear_supported_knobs_still_work``
in ``test_nvfp_dispatch_guards.py``.
"""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction, _qdq
from .utils import prepare_data, calc_snr, calc_cossim


# ---------------------------------------------------------------------------
# QDQ round-trip: BF16 -> NVFP4 -> BF16 (no GEMM).  Each forward/backward
# reduction axis in ``NVFP4LinearFunction`` feeds off of one such round-trip,
# so this is the narrowest test we can write before layering in the matmul.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(128, 64), (4, 128, 64)])
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_per_tensor_scale", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_nvfp4_qdq_roundtrip(
    shape, axis, is_2d_block, use_per_tensor_scale, use_sr,
):
    """Verify precision of a single NVFP4 quant -> dequant round-trip.

    The SNR / cosine-similarity floors are picked to be the loosest value
    still consistent with correct behaviour in every parametrised cell,
    so regressions (e.g. an incorrect scale encoding) show up as threshold
    failures rather than as a subtle degradation that slips through.
    """
    x = prepare_data(shape, torch.bfloat16)

    x_qdq = _qdq(
        x, axis=axis,
        is_2d_block=is_2d_block,
        use_per_tensor_scale=use_per_tensor_scale,
        use_sr=use_sr,
    )

    assert x_qdq.shape == x.shape
    assert x_qdq.dtype == x.dtype

    snr = calc_snr(x, x_qdq)
    cossim = calc_cossim(x, x_qdq)

    print()
    print(tabulate(
        [["SNR (dB)", f"{snr:.2f}"], ["Cosine Sim", f"{cossim:.6f}"]],
        headers=["Metric", "Value"], tablefmt="github",
    ))

    # 1D block: dense per-row/col scales -> tight SNR.  2D block: scales
    # are shared across a block_size × block_size tile so precision drops.
    # 2D + SR is the weakest combination (E4M3 block-scale rounding plus
    # stochastic-rounding variance), so its cossim floor is the loosest.
    min_snr = 8 if not is_2d_block else 5
    min_cossim = 0.99 if not is_2d_block else (0.94 if use_sr else 0.99)
    assert snr > min_snr, f"QDQ SNR too low: {snr:.2f}"
    assert cossim > min_cossim, f"QDQ cosine similarity too low: {cossim:.6f}"


# ---------------------------------------------------------------------------
# Full autograd-function parity test.  Equivalent in structure to MXFP4's
# ``test_mxfp4_linear_autograd_function``: given identical inputs and a
# deterministic MSE target, the NVFP4 forward output and both gradients
# (dX, dW) must track a reference BF16 linear within an SNR threshold.
#
# ``use_hadamard`` / ``use_dge`` are intentionally NOT part of this matrix:
# they are wgrad-only refinements that only meaningfully change the dW
# distribution under long training horizons, and their dispatch-level
# integration is covered separately in ``test_nvfp_dispatch_guards.py``.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(1, 64, 64, 64), (1, 512, 384, 128), (4, 1024, 1024, 2048)])
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_nvfp4_linear_autograd_function(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type,
):
    """Forward + dX + dW must match a BF16 nn.Linear reference in SNR.

    The SR-on-grad path is slightly noisier than RNE (unbiased but higher
    variance), so the threshold is 3 dB with SR and 4 dB without.
    """
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)
    target = prepare_data((B, M, N), data_type)

    # Reference BF16 / FP32 nn.Linear forward + backward.
    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_(); weights.grad.zero_()

    # NVFP4 QDQ path.  This is NOT a shared MXFP4/NVFP4 test; it is specific
    # to NVFP4's autograd function.  We still keep
    # ``use_per_tensor_scale=False`` here on purpose so the parity check stays
    # focused on the shared 6-QDQ linear math (forward + dX + dW).  The
    # tensor-wise scale path is exercised separately in the QDQ round-trip and
    # quantization tests, where it can be isolated without broadening this
    # matrix or coupling the SNR thresholds to an NVFP4-only knob.
    outputs = NVFP4LinearFunction.apply(
        inputs, weights,
        use_2dblock_x, use_2dblock_w, use_sr_grad,
        False,  # use_per_tensor_scale
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs, outputs_ref)
    output_sim = calc_cossim(outputs, outputs_ref)
    dx_snr = calc_snr(inputs.grad, grad_inputs_ref)
    dx_sim = calc_cossim(inputs.grad, grad_inputs_ref)
    dw_snr = calc_snr(weights.grad, grad_weights_ref)
    dw_sim = calc_cossim(weights.grad, grad_weights_ref)

    print()
    print(tabulate(
        [
            ["O",  f"{output_snr:.2f}", f"{output_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}",     f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}",     f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github",
    ))

    min_snr = 3 if use_sr_grad else 4
    assert output_snr > min_snr, f"Output SNR too low: {output_snr:.2f}"
    assert dx_snr     > min_snr, f"dX SNR too low: {dx_snr:.2f}"
    assert dw_snr     > min_snr, f"dW SNR too low: {dw_snr:.2f}"
