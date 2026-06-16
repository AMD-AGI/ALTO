# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch

from alto.kernels.fp4.fp4_common import unwrap_weight_wrapper
from alto.kernels.hadamard_transform import HadamardFactory, HadamardTransform
from alto.kernels.dge import dge_bwd
from .nvfp_quantization import (
    BLOCK_SIZE_DEFAULT,
    _qdq,  # noqa: F401 (re-exported for backward compatibility)
    convert_from_nvfp4,
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
        use_outer_scale: bool,
        hadamard_transform: Optional[HadamardTransform] = None,
        use_dge: bool = False,
    ):
        weight = unwrap_weight_wrapper(weight)
        # Align weight dtype with activation so saved QDQ tensors share a
        # common dtype across forward / backward under AMP autocast (matches
        # the MXFP4 ``original_dtype = x.dtype`` contract).
        if weight.dtype != x.dtype:
            weight = weight.to(dtype=x.dtype)

        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1])

        # Prepare the quantized views used by forward. These are the tensors
        # consumed by the simulated Fprop GEMM.
        x_dq = _qdq(
            x_2d, axis=-1,
            is_2d_block=use_2dblock_x,
            use_outer_scale=use_outer_scale,
        )
        w_dq = _qdq(
            weight, axis=-1,
            is_2d_block=use_2dblock_w,
            use_outer_scale=use_outer_scale,
        )

        y = x_dq @ w_dq.T

        # Prepare separate quantized views for the tensors that participate in
        # the backward reduction dimension. This mirrors how pure FP4 GEMMs
        # would require axis-specific quantized layouts even though the current
        # fallback implementation ultimately multiplies BF16 tensors.
        #
        # For DGE we additionally retain the packed FP4 view + scales of the
        # weight's wgrad-axis quantization so the backward pass can recover the
        # exact FP4 bin each weight fell into, without a second quantize pass.
        #
        # The wgrad reduction-axis view is mathematically different from the
        # fprop view whenever 1D block scaling is used.  Do not silently reuse
        # the fprop view for misaligned shapes: fail fast so callers learn that
        # the current QDQ kernel requires axis-0 dimensions to be block-aligned.
        w_raw_fp4 = None
        w_raw_scales = None
        if not use_2dblock_w:
            torch._check(
                weight.shape[0] % BLOCK_SIZE_DEFAULT == 0,
                lambda: (
                    "NVFP4LinearFunction wgrad-axis weight QDQ requires "
                    f"weight.shape[0] ({weight.shape[0]}) divisible by "
                    f"BLOCK_SIZE_DEFAULT ({BLOCK_SIZE_DEFAULT}) when "
                    "use_2dblock_w=False.  Silent fallback to the fprop "
                    "axis=-1 view is not mathematically valid for 1D blocks."
                ),
            )
            if use_dge:
                w_dq_axis0, w_raw_fp4, w_raw_scales = _qdq(
                    weight, axis=0,
                    is_2d_block=False,
                    use_outer_scale=use_outer_scale,
                    return_raw=True,
                )
            else:
                w_dq_axis0 = _qdq(
                    weight, axis=0,
                    is_2d_block=False,
                    use_outer_scale=use_outer_scale,
                )
        else:
            # 2D block scaling is axis-invariant, so the fprop quantized view is
            # also the correct wgrad-axis view.
            if use_dge:
                # Re-run the fprop-axis QDQ with return_raw to grab the raw FP4
                # view. The dequantized output is identical to w_dq by
                # construction, so we discard it and reuse w_dq for the matmul.
                _, w_raw_fp4, w_raw_scales = _qdq(
                    weight, axis=-1,
                    is_2d_block=use_2dblock_w,
                    use_outer_scale=use_outer_scale,
                    return_raw=True,
                )
            w_dq_axis0 = w_dq

        # Same reasoning for x on the wgrad reduction axis: with 1D block
        # scaling, axis=0 and axis=-1 encode different block layouts, so
        # misaligned shapes must fail fast instead of silently changing the
        # recipe.  Hadamard decorrelates outliers along the batch axis before
        # axis=0 QDQ so the wgrad matmul sees FP4 inputs with more uniform
        # per-block ranges; because the same orthogonal H is applied to
        # grad_output in backward, ``(Hx)ᵀ(Hg) = xᵀg`` is preserved in high
        # precision and only the FP4 rounding sees the decorrelated blocks.
        if not use_2dblock_x:
            torch._check(
                x_2d.shape[0] % BLOCK_SIZE_DEFAULT == 0,
                lambda: (
                    "NVFP4LinearFunction wgrad-axis activation QDQ requires "
                    f"x.shape[0] ({x_2d.shape[0]}) divisible by "
                    f"BLOCK_SIZE_DEFAULT ({BLOCK_SIZE_DEFAULT}) when "
                    "use_2dblock_x=False.  Silent fallback to the fprop "
                    "axis=-1 view is not mathematically valid for 1D blocks."
                ),
            )
            if hadamard_transform is not None:
                x_for_axis0 = hadamard_transform(x_2d, left_mul=True)
            else:
                x_for_axis0 = x_2d
            x_dq_axis0 = _qdq(
                x_for_axis0, axis=0,
                is_2d_block=False,
                use_outer_scale=use_outer_scale,
            )
        else:
            x_dq_axis0 = x_dq

        if use_dge:
            ctx.save_for_backward(x_dq_axis0, w_dq_axis0, w_raw_fp4, w_raw_scales)
        else:
            ctx.save_for_backward(x_dq_axis0, w_dq_axis0)
        ctx.use_2dblock_x = use_2dblock_x
        ctx.use_2dblock_w = use_2dblock_w
        ctx.use_sr_grad = use_sr_grad
        ctx.use_outer_scale = use_outer_scale
        ctx.hadamard_transform = hadamard_transform
        ctx.use_dge = use_dge

        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.use_dge:
            x_dq, w_dq, w_raw_fp4, w_raw_scales = ctx.saved_tensors
        else:
            x_dq, w_dq = ctx.saved_tensors
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])
        # Saved QDQ tensors carry forward's dtype; align them with
        # grad_output to avoid BF16 vs FP32 mismatches under AMP autocast.
        if x_dq.dtype != grad_output.dtype:
            x_dq = x_dq.to(dtype=grad_output.dtype)
        if w_dq.dtype != grad_output.dtype:
            w_dq = w_dq.to(dtype=grad_output.dtype)

        # Gradients are quantized once for each reduction-axis usage:
        #   - axis=-1 feeds the activation-gradient GEMM
        #   - axis=0  feeds the weight-gradient GEMM
        # Stochastic rounding is kept on for gradients because unbiased
        # quantization matters most on the backward path.
        if ctx.use_2dblock_x:
            grad_output_dq = _qdq(
                grad_output, axis=-1,
                is_2d_block=True,
                use_outer_scale=ctx.use_outer_scale,
                use_sr=ctx.use_sr_grad,
            )
            grad_output_m_dq = grad_output_dq
        else:
            grad_output_dq = _qdq(
                grad_output, axis=-1,
                is_2d_block=False,
                use_outer_scale=ctx.use_outer_scale,
                use_sr=ctx.use_sr_grad,
            )
            # Apply the same Hadamard rotation that was used on x in forward,
            # to grad_output before axis=0 QDQ. See the forward-side comment
            # for the invariance argument.
            if ctx.hadamard_transform is not None:
                grad_output_for_m = ctx.hadamard_transform(grad_output, left_mul=True)
            else:
                grad_output_for_m = grad_output
            grad_output_m_dq = _qdq(
                grad_output_for_m, axis=0,
                is_2d_block=False,
                use_outer_scale=ctx.use_outer_scale,
                use_sr=ctx.use_sr_grad,
            )

        grad_inputs = grad_output_dq @ w_dq
        grad_weights = grad_output_m_dq.T @ x_dq

        if ctx.use_dge:
            # Recover the weight's FP4 bin value (no scale applied) by
            # dequantizing the saved packed FP4 view with unit block scales
            # and outer_scale=None.  dge_bwd() then maps each element to a
            # bin-aware gradient magnification factor (clamped to [0, 3]),
            # which scales the STE weight gradient so FP4 "dead weights"
            # pinned near bin boundaries still get non-zero learning signal.
            wgrad_axis_w = 0 if not ctx.use_2dblock_w else -1
            wgrad_is_2d_w = ctx.use_2dblock_w
            unit_scales = torch.ones_like(w_raw_scales)
            w_fp4_values = convert_from_nvfp4(
                w_raw_fp4,
                unit_scales,
                output_dtype=grad_weights.dtype,
                axis=wgrad_axis_w,
                is_2d_block=wgrad_is_2d_w,
                outer_scale=None,
            )
            grad_weights = grad_weights * dge_bwd(w_fp4_values, torch.float4_e2m1fn_x2)

        return (
            grad_inputs.view(*original_shape[:-1], w_dq.shape[-1]),
            grad_weights,
            None, None, None, None, None, None,
        )


def _to_nvfp4_then_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    use_2dblock_x: bool,
    use_2dblock_w: bool,
    use_sr_grad: bool,
    use_outer_scale: bool = False,
    use_hadamard: bool = False,
    use_dge: bool = False,
) -> torch.Tensor:
    """Build the optional Hadamard transform and apply ``NVFP4LinearFunction``.

    Name mirrors ``_to_mxfp4_then_scaled_mm``.  The bias term (if any) is
    applied by the dispatch layer outside this function; this function is
    responsible only for the QDQ + matmul pipeline.
    ``HadamardTransform`` is built lazily here (outside the autograd
    function) so the opaque transform object flows through ``apply`` as a
    non-Tensor argument, matching what MXFP4 does.
    """
    hadamard_transform: Optional[HadamardTransform] = None
    if use_hadamard:
        with torch.no_grad():
            hadamard_transform = HadamardFactory.create_transform(device=b.device)
    return NVFP4LinearFunction.apply(
        a, b,
        use_2dblock_x,
        use_2dblock_w,
        use_sr_grad,
        use_outer_scale,
        hadamard_transform,
        use_dge,
    )
