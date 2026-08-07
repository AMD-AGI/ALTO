# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from .mxfp_quantization import is_cdna4


class MXFP4ForwardOnlyLinearFunction(torch.autograd.Function):
    """Quantize the forward GEMM only; keep the backward in bf16.

    The forward quantizes x and weight to MXFP4 and dequantizes them (QDQ),
    exactly as the standard MXFP4 forward does, then runs ``y = x_dq @ w_dq.T``.
    The backward is a plain bf16 matmul against the saved QDQ operands, with the
    incoming gradient left unquantized:

        grad_x = grad_output @ w_dq
        grad_w = grad_output.T @ x_dq

    This isolates the effect of forward-only quantization: nothing in the gradient
    path is quantized. Unlike ``MXFP4LinearFunction`` this is an additive, separate
    Function (the shared quantized-backward path is left untouched).
    """

    @staticmethod
    def forward(ctx, x, weight, use_2dblock_x, use_2dblock_w):
        assert not is_cdna4(), (
            "MXFP4ForwardOnlyLinearFunction only supports the non-CDNA4 (QDQ) path")
        weight = unwrap_weight_wrapper(weight)

        original_shape = x.shape
        original_dtype = x.dtype
        x = x.reshape(-1, original_shape[-1])

        x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(
            x,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )
        w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(
            weight,
            axis=-1,
            is_2d_block=use_2dblock_w,
        )
        x_dq = torch.ops.torchtitan.convert_from_mxfp4(
            x_mxfp4,
            x_scale,
            original_dtype,
            axis=-1,
            is_2d_block=use_2dblock_x,
        )
        w_dq = torch.ops.torchtitan.convert_from_mxfp4(
            w_mxfp4,
            w_scale,
            original_dtype,
            axis=-1,
            is_2d_block=use_2dblock_w,
        )

        y = x_dq @ w_dq.T

        ctx.save_for_backward(x_dq, w_dq)
        ctx.original_shape = original_shape
        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = ctx.original_shape

        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        grad_inputs = grad_output @ w_dq
        grad_weights = grad_output.T @ x_dq

        return grad_inputs.view(*original_shape[:-1], -1), grad_weights, None, None


def _mxfp4_forward_only(
    a: torch.Tensor,
    b: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_macro_block_scaling: bool = False,
) -> torch.Tensor:
    assert not use_macro_block_scaling, (
        "full_precision_backward does not support two_level_scaling / macro-block scaling")
    return MXFP4ForwardOnlyLinearFunction.apply(a, b, use_2dblock_x, use_2dblock_w)
