# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
from tabulate import tabulate
import torch
from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _qdq,
    _to_nvfp4_then_linear,
)

from .utils import prepare_data, calc_snr, calc_cossim


def _max_abs_error(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x.float() - y.float()).abs().max().item()


def _mean_abs_error(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x.float() - y.float()).abs().mean().item()


def _relative_error(x: torch.Tensor, y: torch.Tensor, threshold: float = 0.0) -> float:
    """Mean relative error, computed only over elements where |x| > threshold.

    When *threshold* > 0, near-zero baseline values are excluded so that the
    metric is not dominated by tiny denominators.
    """
    x_f = x.float()
    y_f = y.float()
    mask = x_f.abs() > threshold
    if mask.sum() == 0:
        return 0.0
    diff = (x_f[mask] - y_f[mask]).abs()
    return (diff / x_f[mask].abs()).mean().item()


# ---------------------------------------------------------------------------
# QDQ round-trip accuracy test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [(128, 64), (2048, 2048), (4, 128, 64)],
)
@pytest.mark.parametrize("axis", [-1, -2])
@pytest.mark.parametrize("is_2d_block", [False, True])
@pytest.mark.parametrize("use_per_tensor_scale", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
def test_nvfp4_qdq_roundtrip(
    shape, axis, is_2d_block, use_per_tensor_scale, use_sr, data_type,
):
    """Verify precision of a single quant -> dequant round-trip (no GEMM)."""
    x = prepare_data(shape, data_type)

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
    mae = _mean_abs_error(x, x_qdq)
    max_ae = _max_abs_error(x, x_qdq)

    print()
    print(
        tabulate(
            [
                ["SNR (dB)", f"{snr:.2f}"],
                ["Cosine Sim", f"{cossim:.6f}"],
                ["Mean Abs Error", f"{mae:.6f}"],
                ["Max Abs Error", f"{max_ae:.4f}"],
            ],
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    min_snr = 8 if not is_2d_block else 5
    # 2D block + SR is the weakest combination — E4M3 block-scale rounding
    # shaves a few more mantissa bits, so the cosine-sim floor here is
    # intentionally the loosest of the four (is_2d_block, use_sr) pairs.
    min_cossim = 0.99 if not is_2d_block else (0.94 if use_sr else 0.99)
    assert snr > min_snr, f"QDQ SNR too low: {snr:.2f}"
    assert cossim > min_cossim, f"QDQ cosine similarity too low: {cossim:.6f}"


# ---------------------------------------------------------------------------
# Forward accuracy test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [(1, 64, 64, 64), (1, 256, 384, 128), (4, 512, 512, 1024)],
)
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_per_tensor_scale", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16])
def test_nvfp4_linear_forward_accuracy(
    shape, use_2dblock_x, use_2dblock_w, use_per_tensor_scale, data_type,
):
    B, M, N, K = shape
    x = prepare_data((B, M, K), data_type)
    w = prepare_data((N, K), data_type)

    y_ref = torch.nn.functional.linear(x, w)

    y_nvfp4 = _to_nvfp4_then_linear(
        x, w,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
        use_per_tensor_scale=use_per_tensor_scale,
    )

    assert y_nvfp4.shape == y_ref.shape
    assert y_nvfp4.dtype == y_ref.dtype

    snr = calc_snr(y_ref, y_nvfp4)
    cossim = calc_cossim(y_ref, y_nvfp4)

    print()
    print(
        tabulate(
            [["Output", f"{snr:.2f}", f"{cossim:.6f}"]],
            headers=["Tensor", "SNR", "Cosine Sim"],
            tablefmt="github",
        )
    )

    assert snr > 5, f"Forward SNR too low: {snr:.2f}"
    assert cossim > 0.95, f"Forward cosine similarity too low: {cossim:.4f}"


# ---------------------------------------------------------------------------
# Autograd accuracy test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [(1, 64, 64, 64), (1, 512, 384, 128), (4, 1024, 1024, 2048)],
)
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_nvfp4_linear_autograd_function(
    shape, use_2dblock_x, use_2dblock_w, use_sr_grad, data_type,
):
    B, M, N, K = shape
    inputs = prepare_data((B, M, K), data_type).requires_grad_(True)
    weights = prepare_data((N, K), data_type).requires_grad_(True)
    target = prepare_data((B, M, N), data_type)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()

    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()

    inputs.grad.zero_()
    weights.grad.zero_()

    outputs = NVFP4LinearFunction.apply(
        inputs, weights,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        False,
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
    print(
        tabulate(
            [
                ["O", f"{output_snr:.2f}", f"{output_sim:.6f}"],
                ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
                ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
            ],
            headers=["Tensor", "SNR", "Cosine Sim"],
            tablefmt="github",
        )
    )

    min_snr = 3 if use_sr_grad else 4
    assert output_snr > min_snr, f"Output SNR too low: {output_snr:.2f}"
    assert dx_snr > min_snr, f"dX SNR too low: {dx_snr:.2f}"
    assert dw_snr > min_snr, f"dW SNR too low: {dw_snr:.2f}"


# ---------------------------------------------------------------------------
# P2P precision comparison: NVFP4 QDQ path vs pure BF16 GEMM
#
# Baseline:  BF16 -> GEMM (accum FP32) -> BF16
# NVFP4:     BF16 -> quant -> FP4 -> dequant -> BF16 -> GEMM -> BF16
#
# This test prints a detailed precision report for every configuration.
# It does NOT assert hard thresholds — its purpose is to quantify the
# exact degradation introduced by the quant -> dequant round-trip.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "shape",
    [(1, 128, 128, 128), (1, 512, 384, 256), (4, 1024, 1024, 2048)],
)
@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_per_tensor_scale", [False])
def test_nvfp4_linear_p2p_vs_bf16(
    shape, use_2dblock_x, use_2dblock_w, use_per_tensor_scale,
):
    """Point-to-point precision comparison between NVFP4 QDQ and pure BF16.

    Uses the same Gaussian + sparse-outlier data as real training scenarios.
    Primary accuracy metric is SNR; auxiliary metrics are printed for analysis.
    """
    B, M, N, K = shape
    dtype = torch.bfloat16

    x = prepare_data((B, M, K), dtype)
    w = prepare_data((N, K), dtype)

    # --- BF16 baseline: bf16 -> gemm (fp32 accum) -> bf16 ---
    y_bf16 = torch.nn.functional.linear(x, w)

    # --- NVFP4 QDQ path: bf16 -> quant -> fp4 -> dequant -> bf16 -> gemm -> bf16 ---
    y_nvfp4 = _to_nvfp4_then_linear(
        x, w,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=False,
        use_per_tensor_scale=use_per_tensor_scale,
    )

    snr = calc_snr(y_bf16, y_nvfp4)
    cossim = calc_cossim(y_bf16, y_nvfp4)
    mae = _mean_abs_error(y_bf16, y_nvfp4)
    max_ae = _max_abs_error(y_bf16, y_nvfp4)
    bf16_amax = y_bf16.float().abs().max().item()
    norm_mae = mae / bf16_amax if bf16_amax > 0 else 0.0

    print()
    print(f"  Shape: B={B}, M={M}, N={N}, K={K}")
    print(f"  2D-block: x={use_2dblock_x}, w={use_2dblock_w}")
    print(f"  per_tensor_scale: {use_per_tensor_scale}")
    print()
    print(
        tabulate(
            [
                ["SNR (dB)", f"{snr:.2f}"],
                ["Cosine Sim", f"{cossim:.6f}"],
                ["Mean Abs Error", f"{mae:.4f}"],
                ["Max Abs Error", f"{max_ae:.4f}"],
                ["Normalized MAE (MAE/max)", f"{norm_mae:.6f}"],
                ["BF16 Output Range", f"[{-bf16_amax:.2f}, {bf16_amax:.2f}]"],
            ],
            headers=["Metric", "Value"],
            tablefmt="github",
        )
    )

    min_snr = 10 if not (use_2dblock_x or use_2dblock_w) else 5
    assert snr > min_snr, f"P2P SNR too low: {snr:.2f}"
