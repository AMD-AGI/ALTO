# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
from tabulate import tabulate

from alto.kernels.blockwise_fp8.blockwise_linear import (
    BlockwiseFP8LinearFunction,
    verify_blockwise_linear_backward,
)


def _make_same_shape_non_contiguous(x: torch.Tensor) -> torch.Tensor:
    # Keep shape unchanged while forcing a strided (non-contiguous) view.
    return x.transpose(-1, -2).contiguous().transpose(-1, -2)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton blockwise FP8 linear tests")
def test_blockwise_fp8_linear_verify_path_small_shape():
    # Cover the verification path used in blockwise_linear.py __main__.
    inputs_ok, weights_ok = verify_blockwise_linear_backward(
        B=1,
        M=128,
        N=128,
        K=128,
        device="cuda",
        atol=2e-1,
        rtol=2e-1,
        compile=False,
    )
    assert inputs_ok
    assert weights_ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton blockwise FP8 linear tests")
@pytest.mark.parametrize("non_contiguous_input", [False, True])
@pytest.mark.parametrize("non_contiguous_weight", [False, True])
def test_blockwise_fp8_linear_contiguous_and_non_contiguous(
    non_contiguous_input: bool,
    non_contiguous_weight: bool,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    b, m, n, k = 1, 128, 128, 128

    x_base = torch.randn((b, m, k), device=device, dtype=dtype)
    w_base = torch.randn((n, k), device=device, dtype=dtype)
    target = torch.randn((b, m, n), device=device, dtype=dtype)

    x_data = _make_same_shape_non_contiguous(x_base) if non_contiguous_input else x_base.contiguous()
    w_data = _make_same_shape_non_contiguous(w_base) if non_contiguous_weight else w_base.contiguous()

    if non_contiguous_input:
        assert not x_data.is_contiguous()
    else:
        assert x_data.is_contiguous()
    if non_contiguous_weight:
        assert not w_data.is_contiguous()
    else:
        assert w_data.is_contiguous()

    x_ref = torch.nn.Parameter(x_data.clone())
    w_ref = torch.nn.Parameter(w_data.clone())
    out_ref = torch.nn.functional.linear(x_ref, w_ref)
    loss_ref = torch.nn.functional.mse_loss(out_ref, target)
    loss_ref.backward()
    dx_ref = x_ref.grad.detach().clone()
    dw_ref = w_ref.grad.detach().clone()

    x_test = torch.nn.Parameter(x_data.clone())
    w_test = torch.nn.Parameter(w_data.clone())
    out_test = BlockwiseFP8LinearFunction.apply(x_test, w_test, 128)
    loss_test = torch.nn.functional.mse_loss(out_test, target)
    loss_test.backward()

    out_snr = _calc_snr(out_test, out_ref)
    dx_snr = _calc_snr(x_test.grad, dx_ref)
    dw_snr = _calc_snr(w_test.grad, dw_ref)
    out_sim = _calc_cossim(out_test, out_ref)
    dx_sim = _calc_cossim(x_test.grad, dx_ref)
    dw_sim = _calc_cossim(w_test.grad, dw_ref)

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
