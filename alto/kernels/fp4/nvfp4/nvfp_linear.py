# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple, Union

import torch

from alto.kernels.hadamard_transform import HadamardFactory, HadamardTransform
from alto.kernels.dge import dge_bwd
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
    return_raw: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Quantize to NVFP4 then immediately dequantize back (QDQ round-trip).

    This helper is the core building block used to emulate the numerical effect
    of a future pure-NVFP4 GEMM path. Even though the current implementation
    executes the final matrix multiplications in BF16, every operand first goes
    through an explicit NVFP4 round-trip so the surrounding autograd code can
    model how a native FP4 kernel would see its inputs.

    When ``return_raw=True``, the packed FP4 data tensor (``uint8``) and the
    per-block scales (``float32`` held as E4M3-round-tripped values) are also
    returned alongside the dequantized BF16 view. This is used by the DGE
    backward path to recover the exact FP4 bin each weight element fell into
    without paying a second quantization pass.
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
    dq = convert_from_nvfp4(
        data_lp,
        scales,
        output_dtype=tensor.dtype,
        block_size=block_size,
        axis=axis,
        is_2d_block=is_2d_block,
        per_tensor_scale=pts,
    )
    if return_raw:
        return dq, data_lp, scales
    return dq


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
        hadamard_transform: Optional[HadamardTransform] = None,
        use_dge: bool = False,
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
        #
        # For DGE we additionally retain the packed FP4 view + scales of the
        # weight's wgrad-axis quantization so the backward pass can recover the
        # exact FP4 bin each weight fell into, without a second quantize pass.
        w_raw_fp4 = None
        w_raw_scales = None
        if not use_2dblock_w:
            if use_dge:
                w_dq_axis0, w_raw_fp4, w_raw_scales = _qdq(
                    weight, axis=0,
                    is_2d_block=False,
                    use_per_tensor_scale=use_per_tensor_scale,
                    return_raw=True,
                )
            else:
                w_dq_axis0 = _qdq(
                    weight, axis=0,
                    is_2d_block=False,
                    use_per_tensor_scale=use_per_tensor_scale,
                )
        else:
            # 2D scaling: fprop quantization already matches wgrad axis layout.
            if use_dge:
                # Re-run the fprop-axis QDQ with return_raw to grab the raw FP4
                # view. The dequantized output is identical to w_dq by
                # construction, so we discard it and reuse w_dq for the matmul.
                _, w_raw_fp4, w_raw_scales = _qdq(
                    weight, axis=-1,
                    is_2d_block=True,
                    use_per_tensor_scale=use_per_tensor_scale,
                    return_raw=True,
                )
            w_dq_axis0 = w_dq

        if not use_2dblock_x:
            # Hadamard decorrelates outliers along the batch axis before axis=0
            # QDQ so the wgrad matmul sees FP4 inputs with more uniform per-
            # block ranges. Because the same orthogonal H is also applied to
            # grad_output in backward, ``(Hx)ᵀ(Hg) = xᵀg`` is preserved in
            # high precision — the rotation only changes how the FP4 rounding
            # clips outliers within a block.
            if hadamard_transform is not None:
                x_for_axis0 = hadamard_transform(x_2d, left_mul=True)
            else:
                x_for_axis0 = x_2d
            x_dq_axis0 = _qdq(
                x_for_axis0, axis=0,
                is_2d_block=False,
                use_per_tensor_scale=use_per_tensor_scale,
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
        ctx.use_per_tensor_scale = use_per_tensor_scale
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
                use_per_tensor_scale=ctx.use_per_tensor_scale,
                use_sr=ctx.use_sr_grad,
            )

        grad_inputs = grad_output_dq @ w_dq
        grad_weights = grad_output_m_dq.T @ x_dq

        if ctx.use_dge:
            # Recover the weight's FP4 bin value (no scale applied) by
            # dequantizing the saved packed FP4 view with unit block scales
            # and PTS=None.  dge_bwd() then maps each element to a bin-aware
            # gradient magnification factor (clamped to [0, 3]), which scales
            # the STE weight gradient so FP4 "dead weights" pinned near bin
            # boundaries still get non-zero learning signal.
            wgrad_axis_w = 0 if not ctx.use_2dblock_w else -1
            wgrad_is_2d_w = ctx.use_2dblock_w
            unit_scales = torch.ones_like(w_raw_scales)
            w_fp4_values = convert_from_nvfp4(
                w_raw_fp4,
                unit_scales,
                output_dtype=grad_weights.dtype,
                axis=wgrad_axis_w,
                is_2d_block=wgrad_is_2d_w,
                per_tensor_scale=None,
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
    use_per_tensor_scale: bool = False,
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
        use_per_tensor_scale,
        hadamard_transform,
        use_dge,
    )
