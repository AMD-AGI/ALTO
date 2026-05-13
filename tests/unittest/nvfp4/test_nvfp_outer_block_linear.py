# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Outer-block integration tests for the NVFP4 Linear autograd path."""

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.testing_utils import check_nvfp4_autograd_snr
from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _to_nvfp4_then_scaled_mm,
)
from .utils import prepare_data, calc_snr, calc_cossim


OUTER_CASES = [
    pytest.param(False, False, id="inner_1x16_outer_1x128"),
    pytest.param(False, True, id="inner_1x16_outer_128x128"),
    pytest.param(True, False, id="inner_16x16_outer_1x128"),
    pytest.param(True, True, id="inner_16x16_outer_128x128"),
]


def _run_linear_autograd_case(
    *,
    shape: tuple[int, int, int, int],
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_outer_2dblock_x: bool,
    use_outer_2dblock_w: bool,
    use_sr_grad: bool,
) -> None:
    B, M, N, K = shape
    dtype = torch.bfloat16
    inputs = prepare_data((B, M, K), dtype).requires_grad_(True)
    weights = prepare_data((N, K), dtype).requires_grad_(True)
    target = prepare_data((B, M, N), dtype)

    outputs_ref = torch.nn.functional.linear(inputs, weights)
    loss_ref = torch.nn.functional.mse_loss(outputs_ref, target)
    loss_ref.backward()
    grad_inputs_ref = inputs.grad.clone()
    grad_weights_ref = weights.grad.clone()
    inputs.grad.zero_()
    weights.grad.zero_()

    outputs = _to_nvfp4_then_scaled_mm(
        inputs,
        weights,
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_outer_scale=False,
        use_outer_block_scale=True,
        use_outer_2dblock_x=use_outer_2dblock_x,
        use_outer_2dblock_w=use_outer_2dblock_w,
        outer_block_size=128,
    )
    loss = torch.nn.functional.mse_loss(outputs, target)
    loss.backward()

    output_snr = calc_snr(outputs_ref, outputs)
    dx_snr = calc_snr(grad_inputs_ref, inputs.grad)
    dw_snr = calc_snr(grad_weights_ref, weights.grad)
    output_sim = calc_cossim(outputs_ref, outputs)
    dx_sim = calc_cossim(grad_inputs_ref, inputs.grad)
    dw_sim = calc_cossim(grad_weights_ref, weights.grad)

    print()
    print(tabulate(
        [
            ["O", f"{output_snr:.2f}", f"{output_sim:.6f}"],
            ["dX", f"{dx_snr:.2f}", f"{dx_sim:.6f}"],
            ["dW", f"{dw_snr:.2f}", f"{dw_sim:.6f}"],
        ],
        headers=["Tensor", "SNR", "Cosine Sim"],
        tablefmt="github",
    ))

    check_nvfp4_autograd_snr(
        {"O": output_snr, "dX": dx_snr, "dW": dw_snr},
        K=K,
        use_sr_grad=use_sr_grad,
        kind="nvfp4_linear",
        context=(
            f"NVFP4Linear outer-block shape={shape} "
            f"inner_x_2d={use_2dblock_x} inner_w_2d={use_2dblock_w} "
            f"outer_x_2d={use_outer_2dblock_x} outer_w_2d={use_outer_2dblock_w}"
        ),
    )


@pytest.mark.parametrize("use_2dblock,use_outer_2dblock", OUTER_CASES)
@pytest.mark.parametrize("use_sr_grad", [False, True])
def test_nvfp4_linear_outer_block_autograd(use_2dblock, use_outer_2dblock, use_sr_grad):
    """Outer-block Linear should satisfy the existing K-aware SNR contract."""
    _run_linear_autograd_case(
        shape=(1, 512, 384, 128),
        use_2dblock_x=use_2dblock,
        use_2dblock_w=use_2dblock,
        use_outer_2dblock_x=use_outer_2dblock,
        use_outer_2dblock_w=use_outer_2dblock,
        use_sr_grad=use_sr_grad,
    )


def test_nvfp4_linear_outer_block_large_k_sr_stress():
    """Keep one K=2048 SR stress case without exploding the test matrix."""
    _run_linear_autograd_case(
        shape=(4, 1024, 1024, 2048),
        use_2dblock_x=True,
        use_2dblock_w=True,
        use_outer_2dblock_x=True,
        use_outer_2dblock_w=True,
        use_sr_grad=True,
    )


def test_nvfp4_linear_outer_block_rejects_dge():
    """Outer-block + DGE needs raw-scale plumbing that is not implemented yet."""
    x = prepare_data((1, 128, 128), torch.bfloat16)
    w = prepare_data((128, 128), torch.bfloat16)
    with pytest.raises(RuntimeError, match="does not yet support DGE"):
        NVFP4LinearFunction.apply(
            x,
            w,
            False,
            False,
            False,
            False,
            None,
            True,
            True,
            False,
            False,
            128,
        )
