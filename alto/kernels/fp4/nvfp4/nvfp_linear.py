# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch

from .nvfp_quantization import (
    convert_to_nvfp4,
    convert_from_nvfp4,
)


def _qdq(
    tensor: torch.Tensor,
    *,
    axis: int,
    is_2d_block: bool,
    use_per_tensor_scale: bool,
    use_sr: bool = False,
    block_size: int = 16,
) -> torch.Tensor:
    """Quantize to NVFP4 then immediately dequantize back (QDQ round-trip).

    This helper is the core building block used to emulate the numerical effect
    of a future pure-NVFP4 GEMM path. Even though the current implementation
    executes the final matrix multiplications in BF16, every operand first goes
    through an explicit NVFP4 round-trip so the surrounding autograd code can
    model how a native FP4 kernel would see its inputs.
    """
    pts = (
        torch.empty(1, dtype=torch.float32, device=tensor.device)
        if use_per_tensor_scale
        else None
    )
    data_lp, scales = convert_to_nvfp4(
        tensor,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        per_tensor_scale=pts,
        # With pts provided, refresh it in-place from tensor.amax(); without
        # pts, skip tensor-wise scaling entirely.
        update_per_tensor_scale=use_per_tensor_scale,
        use_sr=use_sr,
    )
    return convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=tensor.dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        per_tensor_scale=pts,
    )


@torch.compiler.allow_in_graph
class NVFP4LinearFunction(torch.autograd.Function):
    """Custom autograd function for NVFP4 low-precision linear.

    Since there is no FP4 GEMM hardware support for NVFP4, the data path is:

        BF16 -> quant -> FP4 -> dequant -> BF16 -> GEMM -> FP32 -> BF16

    This is equivalent to inserting a QDQ noise layer around every operand
    before the standard BF16 matmul.

    Important implementation note:
    this class is written to mirror the operand preparation pattern that would
    be required by a future pure NVFP4 training kernel, not to minimize the
    number of software-emulated QDQ steps on the current BF16 fallback path.

    Concretely, the implementation keeps separate quantized views for the
    forward-reduction dimension and the backward-reduction dimension whenever
    necessary. This yields a 6-QDQ style simulation:

    1. QDQ activations for forward
    2. QDQ weights for forward
    3. QDQ weights again for the backward reduction axis
    4. QDQ activations again for the backward reduction axis
    5. QDQ output gradients for activation-gradient GEMM
    6. QDQ output gradients for weight-gradient GEMM

    The extra axis-specific QDQ states are intentional here because this module
    is meant to approximate the quantized inputs consumed by pure FP4 GEMMs in
    both forward and backward, rather than serve as the numerically best
    software-only recipe for the current AMD path.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        use_2dblock_x: bool,
        use_2dblock_w: bool,
        use_sr_grad: bool,
        use_per_tensor_scale: bool,
    ):
        # Explicitly unwrap any tensor-subclass weight (e.g.
        # ``NVFP4TrainingWeightWrapperTensor`` from the dispatch layer, or a
        # DTensor under FSDP2) down to a plain ``torch.Tensor`` at the
        # autograd-function boundary.  Downstream aten ops would also
        # unwrap on their own via ``__torch_dispatch__``, but doing it here
        # once keeps the autograd tape + ``ctx.save_for_backward`` on
        # plain tensors and avoids paying a pytree walk on every op call.
        if type(weight) is not torch.Tensor:
            if hasattr(weight, "_data"):
                weight = weight._data
            else:
                weight = weight.as_subclass(torch.Tensor)

        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        # Prepare the quantized views used by forward. These are the tensors
        # consumed by the simulated Fprop GEMM.
        x_dq = _qdq(
            x_2d, axis=-1,
            is_2d_block=use_2dblock_x,
            use_per_tensor_scale=use_per_tensor_scale,
        )
        w_dq = _qdq(
            weight, axis=-1,
            is_2d_block=use_2dblock_w,
            use_per_tensor_scale=use_per_tensor_scale,
        )

        y = x_dq @ w_dq.T

        # Prepare separate quantized views for the tensors that participate in
        # the backward reduction dimension. This mirrors how pure FP4 GEMMs
        # would require axis-specific quantized layouts even though the current
        # fallback implementation ultimately multiplies BF16 tensors.
        if not use_2dblock_w:
            w_dq_axis0 = _qdq(
                weight, axis=0,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
            )
        else:
            w_dq_axis0 = w_dq

        if not use_2dblock_x:
            x_dq_axis0 = _qdq(
                x_2d, axis=0,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
            )
        else:
            x_dq_axis0 = x_dq

        ctx.save_for_backward(x_dq_axis0, w_dq_axis0)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.use_sr_grad = use_sr_grad
        ctx.use_per_tensor_scale = use_per_tensor_scale

        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])

        # Gradients are quantized once for each reduction-axis usage:
        #   - axis=-1 feeds the activation-gradient GEMM
        #   - axis=0  feeds the weight-gradient GEMM
        # Stochastic rounding is kept on for gradients because unbiased
        # quantization matters most on the backward path.
        if ctx.use_2dblock_x:
            grad_output_dq = _qdq(
                grad_output, axis=-1,
                is_2d_block=True,
                use_per_tensor_scale=ctx.use_per_tensor_scale,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_m_dq = grad_output_dq
        else:
            grad_output_dq = _qdq(
                grad_output, axis=-1,
                is_2d_block=False,
                use_per_tensor_scale=ctx.use_per_tensor_scale,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_m_dq = _qdq(
                grad_output, axis=0,
                is_2d_block=False,
                use_per_tensor_scale=ctx.use_per_tensor_scale,
                use_sr=ctx.use_sr_grad,
            )

        grad_inputs = grad_output_dq @ w_dq
        grad_weights = grad_output_m_dq.T @ x_dq

        return (
            grad_inputs.view(*original_shape[:-1], -1),
            grad_weights,
            None, None, None, None,
        )


def _to_nvfp4_then_linear(
    a: torch.Tensor,
    b: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_per_tensor_scale: bool = False,
) -> torch.Tensor:
    """Public entry-point: NVFP4 QDQ + BF16 linear.

    The public API keeps the full parameter surface because the function is
    intended to model future pure-NVFP4 linear behavior, including cases where
    forward and backward consume different quantized views of the same tensor.
    """
    return NVFP4LinearFunction.apply(
        a, b,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_per_tensor_scale,
    )
