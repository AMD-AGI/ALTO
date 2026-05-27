# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""ALTO-side MXFP4 + per-slot Hadamard autograd Function.

Port of AdaHOP's ``MXFP4LinearFunction`` adapted for ALTO's tensor-subclass
wrapper design: modes are passed as ``apply()`` arguments instead of looked
up from a global FQN-keyed registry.

Reference (AdaHOP, BSD-3):
  3rdparty/adahop/torchtitan/experiments/kernels/mxfp4/mxfp_linear.py:206-635

Key differences from AdaHOP's version:
  * ``layer_name`` argument removed; ``forward_y_mode``, ``backward_gx_mode``,
    ``backward_gw_mode`` are explicit ``apply()`` arguments and ride on
    ``ctx`` for backward.
  * Calibration-mode globals (``is_calibration_mode``, ``store_detected_pattern``)
    removed; calibration is done by the wrapper's ``__torch_function__``
    via a callback installed by the modifier.
  * Tensor-capture mode removed (debug-only feature, not needed for MVP).
  * ``iht_quantization`` (AdaHOP's fused Hadamard+MXFP4 Triton kernel) is
    not ported; ``hadamard`` mode falls back to the equivalent two-step
    ``hadamard_transform`` → ``convert_to_mxfp4``. Slower than AdaHOP but
    numerically equivalent for round-to-nearest. SR composability documented
    inline.
  * Outlier-extract modes (``inner_outlier_extract_*``, ``outer_outlier_extract_*``)
    are guarded by ``assert_mode_supported`` and raise ``NotImplementedError``
    at apply time. Add them when calibration starts producing those modes.
  * Calls ALTO's already-registered ``torch.ops.torchtitan.{convert_to_mxfp4,
    blockwise_mxfp4_gemm}`` — no new op registrations, no collision with
    ALTO's own ``alto/kernels/fp4/mxfp4/mxfp_linear.py``.
"""

from typing import Optional

import torch

from .transform_mode import TransformMode, assert_mode_supported

HadamardTransformType = "HadamardTransform"  # type-hint placeholder; avoid hard import


@torch.compiler.allow_in_graph
class MXFP4AdaHOPLinearFunction(torch.autograd.Function):
    """MXFP4 linear with per-slot Hadamard, modes baked into ``apply`` args.

    The 3 slots and their AdaHOP names:
      * ``forward_y``    — applied to the forward ``y = x @ wᵀ`` computation
      * ``backward_gx``  — applied to ``grad_inputs = grad_output @ w``
      * ``backward_gw``  — applied to ``grad_weights = grad_outputᵀ @ x``
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        use_sr_grad: bool,
        hadamard_transform: Optional["HadamardTransformType"],
        forward_y_mode: TransformMode,
        backward_gx_mode: TransformMode,
        backward_gw_mode: TransformMode,
    ) -> torch.Tensor:
        assert_mode_supported(forward_y_mode, "forward_y")
        assert_mode_supported(backward_gx_mode, "backward_gx")
        assert_mode_supported(backward_gw_mode, "backward_gw")

        original_shape = x.shape
        original_dtype = x.dtype
        x = x.reshape(-1, original_shape[-1])

        # ---- forward: y = x @ wᵀ under forward_y_mode -----------------------
        if forward_y_mode == "full_precision":
            y = x @ weight.T
            # Still need to prepare x/w for backward branches below.
            # full_precision in forward does NOT short-circuit backward prep.
            x_mxfp4_fwd, x_scale_fwd, w_mxfp4_fwd, w_scale_fwd = None, None, None, None
        else:
            if forward_y_mode == "hadamard":
                assert hadamard_transform is not None, ("forward_y_mode=hadamard requires a hadamard_transform")
                # Unfused fallback for AdaHOP's iht_quantization(x, left_mul=False):
                # right-Hadamard then axis=-1 quantize.
                x_for_y = hadamard_transform(x, left_mul=False)
                w_for_y = hadamard_transform(weight, left_mul=False)
            elif forward_y_mode == "outer_hadamard":
                assert hadamard_transform is not None, ("forward_y_mode=outer_hadamard requires a hadamard_transform")
                # left-multiply pre-transform; the outer-HT is applied after the GEMM.
                x_for_y = hadamard_transform(x, left_mul=True)
                w_for_y = hadamard_transform(weight, left_mul=True)
            else:  # "none"
                x_for_y = x
                w_for_y = weight

            x_mxfp4_fwd, x_scale_fwd = torch.ops.torchtitan.convert_to_mxfp4(
                x_for_y,
                axis=-1,
                is_2d_block=False,
            )
            w_mxfp4_fwd, w_scale_fwd = torch.ops.torchtitan.convert_to_mxfp4(
                w_for_y,
                axis=-1,
                is_2d_block=False,
            )
            y = torch.ops.torchtitan.blockwise_mxfp4_gemm(
                x_mxfp4_fwd,
                x_scale_fwd,
                w_mxfp4_fwd,
                w_scale_fwd,
                trans_b=True,
                output_dtype=original_dtype,
            )
            if forward_y_mode == "outer_hadamard":
                y = hadamard_transform(hadamard_transform(y, left_mul=True))

        # ---- prepare w for backward_gx (grad_output @ w) --------------------
        w_mxfp4_bx, w_scale_bx = _prepare_operand_for_backward(
            tensor=weight,
            mode=backward_gx_mode,
            hadamard_transform=hadamard_transform,
        )

        # ---- prepare x for backward_gw (grad_outputᵀ @ x) -------------------
        x_mxfp4_bw, x_scale_bw = _prepare_operand_for_backward(
            tensor=x,
            mode=backward_gw_mode,
            hadamard_transform=hadamard_transform,
        )

        ctx.save_for_backward(x_mxfp4_bw, x_scale_bw, w_mxfp4_bx, w_scale_bx)
        ctx.original_dtype = original_dtype
        ctx.use_sr_grad = use_sr_grad
        ctx.hadamard_transform = hadamard_transform
        ctx.forward_y_mode = forward_y_mode
        ctx.backward_gx_mode = backward_gx_mode
        ctx.backward_gw_mode = backward_gw_mode

        return y.view(*original_shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        original_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, original_shape[-1])

        x_mxfp4_bw, x_scale_bw, w_mxfp4_bx, w_scale_bx = ctx.saved_tensors

        grad_inputs = _backward_gx(
            grad_output=grad_output,
            w_mxfp4=w_mxfp4_bx,
            w_scale=w_scale_bx,
            mode=ctx.backward_gx_mode,
            hadamard_transform=ctx.hadamard_transform,
            use_sr_grad=ctx.use_sr_grad,
            original_dtype=ctx.original_dtype,
        )

        grad_weights = _backward_gw(
            grad_output=grad_output,
            x_mxfp4=x_mxfp4_bw,
            x_scale=x_scale_bw,
            mode=ctx.backward_gw_mode,
            hadamard_transform=ctx.hadamard_transform,
            use_sr_grad=ctx.use_sr_grad,
            original_dtype=ctx.original_dtype,
        )

        # forward signature has 7 inputs (x, weight, use_sr_grad, hadamard_transform,
        # forward_y_mode, backward_gx_mode, backward_gw_mode); return 7 grads with
        # only the first two non-None.
        return (
            grad_inputs.view(*original_shape[:-1], -1),
            grad_weights,
            None,
            None,
            None,
            None,
            None,
        )


def _prepare_operand_for_backward(
    *,
    tensor: torch.Tensor,
    mode: TransformMode,
    hadamard_transform,
):
    """Pre-rotate and quantize ``tensor`` for the backward GEMM.

    Returns ``(packed_or_dense, scale_or_None)``. For ``full_precision``,
    returns the original tensor and ``None`` (the consumer detects ``None``
    scale to mean "no quant").
    """
    if mode == "full_precision":
        return tensor, None
    if mode == "hadamard":
        assert hadamard_transform is not None, "hadamard mode requires a hadamard_transform"
        prepared = hadamard_transform(tensor, left_mul=True)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None, "outer_hadamard requires a hadamard_transform"
        prepared = hadamard_transform(tensor)
    else:  # "none"
        prepared = tensor

    t_mxfp4, t_scale = torch.ops.torchtitan.convert_to_mxfp4(prepared, axis=0, is_2d_block=False)
    return t_mxfp4, t_scale


def _backward_gx(
    *,
    grad_output: torch.Tensor,
    w_mxfp4: torch.Tensor,
    w_scale: Optional[torch.Tensor],
    mode: TransformMode,
    hadamard_transform,
    use_sr_grad: bool,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """``grad_inputs = grad_output @ w`` under ``backward_gx_mode``."""
    if mode == "full_precision":
        # w_mxfp4 is actually the unquantized weight here.
        return grad_output @ w_mxfp4

    if mode == "hadamard":
        assert hadamard_transform is not None
        g_for_gx = hadamard_transform(grad_output, left_mul=False)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None
        g_for_gx = hadamard_transform(grad_output, left_mul=True)
    else:  # "none"
        g_for_gx = grad_output

    g_mxfp4, g_scale = torch.ops.torchtitan.convert_to_mxfp4(
        g_for_gx,
        axis=-1,
        use_sr=use_sr_grad,
        is_2d_block=False,
    )
    grad_inputs = torch.ops.torchtitan.blockwise_mxfp4_gemm(
        g_mxfp4,
        g_scale,
        w_mxfp4,
        w_scale,
        output_dtype=original_dtype,
    )
    if mode == "outer_hadamard":
        grad_inputs = hadamard_transform(hadamard_transform(grad_inputs, left_mul=True))
    return grad_inputs


def _backward_gw(
    *,
    grad_output: torch.Tensor,
    x_mxfp4: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    mode: TransformMode,
    hadamard_transform,
    use_sr_grad: bool,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """``grad_weights = grad_outputᵀ @ x`` under ``backward_gw_mode``."""
    if mode == "full_precision":
        # x_mxfp4 is actually the unquantized x here.
        return grad_output.T @ x_mxfp4

    if mode == "hadamard":
        assert hadamard_transform is not None
        g_for_gw = hadamard_transform(grad_output, left_mul=True)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None
        g_for_gw = hadamard_transform(grad_output)
    else:  # "none"
        g_for_gw = grad_output

    g_mxfp4_m, g_scale_m = torch.ops.torchtitan.convert_to_mxfp4(
        g_for_gw,
        axis=0,
        use_sr=use_sr_grad,
        is_2d_block=False,
    )
    grad_weights = torch.ops.torchtitan.blockwise_mxfp4_gemm(
        g_mxfp4_m,
        g_scale_m,
        x_mxfp4,
        x_scale,
        trans_a=True,
        output_dtype=original_dtype,
    )
    if mode == "outer_hadamard":
        grad_weights = hadamard_transform(hadamard_transform(grad_weights, left_mul=True))
    return grad_weights
