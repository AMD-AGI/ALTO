# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""ALTO-side MXFP4 + per-slot Hadamard autograd Function.

Port of AdaHOP's ``MXFP4LinearFunction`` adapted for ALTO's tensor-subclass
wrapper design: modes are passed as ``apply()`` arguments instead of looked
up from a global FQN-keyed registry.

Reference (AdaHOP, BSD-3):
  3rdparty/adahop/torchtitan/experiments/kernels/mxfp4/mxfp_linear.py:206-635

Supported modes per slot:
  * ``none``
  * ``hadamard`` / ``outer_hadamard``
  * ``inner_outlier_extract_left`` / ``inner_outlier_extract_right``
  * ``full_precision``

Notes vs AdaHOP's version:
  * ``layer_name`` removed; modes are explicit ``apply()`` args.
  * Calibration/tensor-capture globals removed; calibration runs via the
    wrapper's ``__torch_function__`` callback installed by the modifier.
  * ``hadamard`` mode uses AdaHOP's fused ``iht_quantization`` kernel
    (loaded via the bridge); when SR is requested it composes through.
  * Outlier-extract paths use ``inner_outlier_extract_{left,right}_cdna4``
    from AdaHOP's submodule — those kernels self-gate cdna3 (gfx942) via
    PyTorch fallbacks (see outlier_extract.py:165, :323).
  * Forward ``inner_outlier_extract_left`` saves the BF16 weight to
    ``ctx`` for backward (the kernel takes B as BF16 internally).
  * Backward ``inner_outlier_extract_left`` for ``backward_gw`` falls
    back to plain ``hadamard`` semantics (matches AdaHOP's note at
    mxfp_linear.py:560-575).
"""

from typing import Optional, Tuple

import torch

from alto.kernels.fp4.mxfp4.mxfp_quantization import is_cdna4
from .transform_mode import TransformMode, assert_mode_supported

HadamardTransformType = "HadamardTransform"  # type-hint placeholder; avoid hard import


def _blockwise_mxfp4_gemm_or_dequant(
    a_mxfp4: torch.Tensor,
    a_scale: torch.Tensor,
    b_mxfp4: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    trans_a: bool = False,
    trans_b: bool = False,
) -> torch.Tensor:
    """Triton MXFP4 GEMM on cdna4, dequant-then-bf16-matmul fallback on cdna3.
    Mirrors ``alto/kernels/fp4/mxfp4/mxfp_linear.py:321-353``."""
    if is_cdna4():
        return torch.ops.torchtitan.blockwise_mxfp4_gemm(
            a_mxfp4,
            a_scale,
            b_mxfp4,
            b_scale,
            trans_a=trans_a,
            trans_b=trans_b,
            output_dtype=output_dtype,
        )
    a_dq = torch.ops.torchtitan.convert_from_mxfp4(
        a_mxfp4, a_scale, output_dtype, axis=-1, is_2d_block=False,
    )
    b_dq = torch.ops.torchtitan.convert_from_mxfp4(
        b_mxfp4, b_scale, output_dtype, axis=-1, is_2d_block=False,
    )
    if trans_a:
        a_dq = a_dq.T
    if trans_b:
        b_dq = b_dq.T
    return a_dq @ b_dq


@torch.compiler.allow_in_graph
class MXFP4AdaHOPLinearFunction(torch.autograd.Function):
    """MXFP4 linear with per-slot Hadamard / outlier-extract, modes baked in.

    Slots:
      * ``forward_y``    — ``y = x @ wᵀ``
      * ``backward_gx``  — ``grad_inputs = grad_output @ w``
      * ``backward_gw``  — ``grad_weights = grad_outputᵀ @ x``
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

        # ---- forward: y = x @ wᵀ ------------------------------------------
        y = _forward_y(
            x=x,
            weight=weight,
            mode=forward_y_mode,
            hadamard_transform=hadamard_transform,
            original_dtype=original_dtype,
        )

        # ---- prep operands for the two backward GEMMs ---------------------
        w_bx, w_scale_bx, w_outlier_compact_bx, w_outlier_indices_bx = _prep_w_for_gx(
            weight=weight,
            mode=backward_gx_mode,
            hadamard_transform=hadamard_transform,
        )
        x_bw, x_scale_bw, x_outlier_compact_bw, x_outlier_indices_bw = _prep_x_for_gw(
            x=x,
            mode=backward_gw_mode,
            hadamard_transform=hadamard_transform,
            use_sr_grad=use_sr_grad,
        )

        ctx.save_for_backward(
            x_bw, x_scale_bw, w_bx, w_scale_bx,
            x_outlier_compact_bw, x_outlier_indices_bw,
            w_outlier_compact_bx, w_outlier_indices_bx,
        )
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

        (x_bw, x_scale_bw, w_bx, w_scale_bx,
         x_outlier_compact_bw, x_outlier_indices_bw,
         w_outlier_compact_bx, w_outlier_indices_bx) = ctx.saved_tensors

        grad_inputs = _backward_gx(
            grad_output=grad_output,
            w=w_bx,
            w_scale=w_scale_bx,
            w_outlier_compact=w_outlier_compact_bx,
            w_outlier_indices=w_outlier_indices_bx,
            mode=ctx.backward_gx_mode,
            hadamard_transform=ctx.hadamard_transform,
            use_sr_grad=ctx.use_sr_grad,
            original_dtype=ctx.original_dtype,
        )

        grad_weights = _backward_gw(
            grad_output=grad_output,
            x=x_bw,
            x_scale=x_scale_bw,
            x_outlier_compact=x_outlier_compact_bw,
            x_outlier_indices=x_outlier_indices_bw,
            mode=ctx.backward_gw_mode,
            hadamard_transform=ctx.hadamard_transform,
            use_sr_grad=ctx.use_sr_grad,
            original_dtype=ctx.original_dtype,
        )

        # forward has 7 inputs; only the first two get grads.
        return (
            grad_inputs.view(*original_shape[:-1], -1),
            grad_weights,
            None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Forward y
# ---------------------------------------------------------------------------

def _forward_y(
    *,
    x: torch.Tensor,
    weight: torch.Tensor,
    mode: TransformMode,
    hadamard_transform,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute ``y = x @ wᵀ`` under ``forward_y_mode``."""
    if mode == "full_precision":
        return x @ weight.T

    if mode == "inner_outlier_extract_left":
        # Extract row outliers from x, quantize the clean part, run the
        # AdaHOP outlier-aware GEMM. Weight stays BF16; the kernel applies
        # its own HT+quant to it internally.
        from alto._adahop_bridge import (
            iht_quantization,
            inner_outlier_extract_left_cdna4,
            prepare_outlier_clean_row,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        x_clean, x_outlier_compact, x_outlier_indices = prepare_outlier_clean_row(x, k=OUTLIER_K)
        x_clean_mxfp4, x_clean_scale = iht_quantization(x_clean, left_mul=False)
        return inner_outlier_extract_left_cdna4(
            x_clean_mxfp4,
            x_clean_scale,
            x_outlier_compact,
            x_outlier_indices,
            weight,
            hadamard_transform,
            trans_a=False,
            trans_b=True,
            k=OUTLIER_K,
            original_dtype=original_dtype,
        )

    if mode == "inner_outlier_extract_right":
        # Extract row outliers from weight (== col outliers in wᵀ),
        # quantize the clean weight, run the AdaHOP outlier-aware GEMM.
        # x stays BF16; the kernel handles it.
        from alto._adahop_bridge import (
            iht_quantization,
            inner_outlier_extract_right_cdna4,
            prepare_outlier_clean_row,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        w_clean, w_outlier_compact, w_outlier_indices = prepare_outlier_clean_row(weight, k=OUTLIER_K)
        w_clean_mxfp4, w_clean_scale = iht_quantization(w_clean, left_mul=False)
        return inner_outlier_extract_right_cdna4(
            x,
            w_clean_mxfp4,
            w_clean_scale,
            w_outlier_compact,
            w_outlier_indices,
            hadamard_transform,
            trans_a=False,
            trans_b=True,
            k=OUTLIER_K,
            original_dtype=original_dtype,
        )

    # hadamard / outer_hadamard / none — two-step convert+GEMM.
    if mode == "hadamard":
        assert hadamard_transform is not None
        x_for_y = hadamard_transform(x, left_mul=False)
        w_for_y = hadamard_transform(weight, left_mul=False)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None
        x_for_y = hadamard_transform(x, left_mul=True)
        w_for_y = hadamard_transform(weight, left_mul=True)
    else:  # "none"
        x_for_y = x
        w_for_y = weight

    x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(
        x_for_y, axis=-1, is_2d_block=False,
    )
    w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(
        w_for_y, axis=-1, is_2d_block=False,
    )
    y = _blockwise_mxfp4_gemm_or_dequant(
        x_mxfp4, x_scale, w_mxfp4, w_scale,
        trans_b=True, output_dtype=original_dtype,
    )
    if mode == "outer_hadamard":
        y = hadamard_transform(hadamard_transform(y, left_mul=True))
    return y


# ---------------------------------------------------------------------------
# Backward operand preparation (runs in forward, saves into ctx)
# ---------------------------------------------------------------------------

def _prep_w_for_gx(
    *,
    weight: torch.Tensor,
    mode: TransformMode,
    hadamard_transform,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare weight for ``backward_gx`` (``grad_output @ w``).

    Returns ``(w_data, w_scale, w_outlier_compact, w_outlier_indices)``.
    Last three are ``None`` unless mode needs them.
    """
    if mode == "full_precision":
        return weight, None, None, None

    if mode == "inner_outlier_extract_left":
        # Outliers will be extracted from grad_output at backward time;
        # weight is saved as BF16 (kernel handles HT+quant internally).
        return weight, None, None, None

    if mode == "inner_outlier_extract_right":
        # Extract column outliers from weight now; quantize the clean part.
        from alto._adahop_bridge import (
            iht_quantization,
            prepare_outlier_clean_column,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        w_clean, w_outlier_compact, w_outlier_indices = prepare_outlier_clean_column(weight, k=OUTLIER_K)
        w_mxfp4, w_scale = iht_quantization(w_clean, left_mul=True)
        return w_mxfp4, w_scale, w_outlier_compact, w_outlier_indices

    # hadamard / outer_hadamard / none — single-tensor convert path.
    if mode == "hadamard":
        from alto._adahop_bridge import iht_quantization
        assert hadamard_transform is not None
        w_mxfp4, w_scale = iht_quantization(weight, left_mul=True)
        return w_mxfp4, w_scale, None, None
    if mode == "outer_hadamard":
        assert hadamard_transform is not None
        prepared = hadamard_transform(weight)
        w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(prepared, axis=0, is_2d_block=False)
        return w_mxfp4, w_scale, None, None
    # "none"
    w_mxfp4, w_scale = torch.ops.torchtitan.convert_to_mxfp4(weight, axis=0, is_2d_block=False)
    return w_mxfp4, w_scale, None, None


def _prep_x_for_gw(
    *,
    x: torch.Tensor,
    mode: TransformMode,
    hadamard_transform,
    use_sr_grad: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Prepare x for ``backward_gw`` (``grad_outputᵀ @ x``).

    Returns ``(x_data, x_scale, x_outlier_compact, x_outlier_indices)``.
    Last three are ``None`` unless mode needs them.

    Note: ``inner_outlier_extract_left`` for ``backward_gw`` is identical
    to plain ``hadamard`` (per AdaHOP mxfp_linear.py:560-575) — left OE
    can't compose with backward_gw because x needs to be saved as MXFP4
    for memory, conflicting with the BF16 requirement of the left kernel.
    """
    if mode == "full_precision":
        return x, None, None, None

    if mode in ("hadamard", "inner_outlier_extract_left"):
        from alto._adahop_bridge import iht_quantization
        assert hadamard_transform is not None
        x_mxfp4, x_scale = iht_quantization(x, left_mul=True)
        return x_mxfp4, x_scale, None, None

    if mode == "inner_outlier_extract_right":
        from alto._adahop_bridge import (
            iht_quantization,
            prepare_outlier_clean_column,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        x_clean, x_outlier_compact, x_outlier_indices = prepare_outlier_clean_column(x, k=OUTLIER_K)
        x_mxfp4, x_scale = iht_quantization(x_clean, left_mul=True)
        return x_mxfp4, x_scale, x_outlier_compact, x_outlier_indices

    if mode == "outer_hadamard":
        assert hadamard_transform is not None
        prepared = hadamard_transform(x)
        x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(prepared, axis=0, is_2d_block=False)
        return x_mxfp4, x_scale, None, None

    # "none"
    x_mxfp4, x_scale = torch.ops.torchtitan.convert_to_mxfp4(x, axis=0, is_2d_block=False)
    return x_mxfp4, x_scale, None, None


# ---------------------------------------------------------------------------
# Backward compute
# ---------------------------------------------------------------------------

def _backward_gx(
    *,
    grad_output: torch.Tensor,
    w: torch.Tensor,
    w_scale: Optional[torch.Tensor],
    w_outlier_compact: Optional[torch.Tensor],
    w_outlier_indices: Optional[torch.Tensor],
    mode: TransformMode,
    hadamard_transform,
    use_sr_grad: bool,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """``grad_inputs = grad_output @ w`` under ``backward_gx_mode``."""
    if mode == "full_precision":
        # w is the unquantized weight here.
        return grad_output @ w

    if mode == "inner_outlier_extract_left":
        # Extract row outliers from grad_output; w is BF16 in `w`.
        from alto._adahop_bridge import (
            iht_quantization,
            inner_outlier_extract_left_cdna4,
            prepare_outlier_clean_row,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        g_clean, g_outlier_compact, g_outlier_indices = prepare_outlier_clean_row(grad_output, k=OUTLIER_K)
        g_clean_mxfp4, g_clean_scale = iht_quantization(
            g_clean, left_mul=False, use_sr=use_sr_grad,
        )
        return inner_outlier_extract_left_cdna4(
            g_clean_mxfp4,
            g_clean_scale,
            g_outlier_compact,
            g_outlier_indices,
            w,  # BF16 weight
            hadamard_transform,
            trans_a=False,
            trans_b=False,
            k=OUTLIER_K,
            original_dtype=original_dtype,
        )

    if mode == "inner_outlier_extract_right":
        # Pre-extracted column outliers from weight (saved at forward).
        from alto._adahop_bridge import (
            inner_outlier_extract_right_cdna4,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        return inner_outlier_extract_right_cdna4(
            grad_output,
            w,
            w_scale,
            w_outlier_compact,
            w_outlier_indices,
            hadamard_transform,
            trans_a=False,
            trans_b=False,
            k=OUTLIER_K,
            original_dtype=original_dtype,
        )

    # hadamard / outer_hadamard / none
    if mode == "hadamard":
        from alto._adahop_bridge import iht_quantization
        assert hadamard_transform is not None
        g_mxfp4, g_scale = iht_quantization(grad_output, left_mul=False, use_sr=use_sr_grad)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None
        g_transformed = hadamard_transform(grad_output, left_mul=True)
        g_mxfp4, g_scale = torch.ops.torchtitan.convert_to_mxfp4(
            g_transformed, axis=-1, use_sr=use_sr_grad, is_2d_block=False,
        )
    else:  # "none"
        g_mxfp4, g_scale = torch.ops.torchtitan.convert_to_mxfp4(
            grad_output, axis=-1, use_sr=use_sr_grad, is_2d_block=False,
        )

    grad_inputs = _blockwise_mxfp4_gemm_or_dequant(
        g_mxfp4, g_scale, w, w_scale,
        output_dtype=original_dtype,
    )
    if mode == "outer_hadamard":
        grad_inputs = hadamard_transform(hadamard_transform(grad_inputs, left_mul=True))
    return grad_inputs


def _backward_gw(
    *,
    grad_output: torch.Tensor,
    x: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    x_outlier_compact: Optional[torch.Tensor],
    x_outlier_indices: Optional[torch.Tensor],
    mode: TransformMode,
    hadamard_transform,
    use_sr_grad: bool,
    original_dtype: torch.dtype,
) -> torch.Tensor:
    """``grad_weights = grad_outputᵀ @ x`` under ``backward_gw_mode``."""
    if mode == "full_precision":
        # x is the unquantized input here.
        return grad_output.T @ x

    if mode == "inner_outlier_extract_right":
        from alto._adahop_bridge import (
            inner_outlier_extract_right_cdna4,
            OUTLIER_K,
        )
        assert hadamard_transform is not None
        return inner_outlier_extract_right_cdna4(
            grad_output,
            x,
            x_scale,
            x_outlier_compact,
            x_outlier_indices,
            hadamard_transform,
            trans_a=True,
            trans_b=False,
            k=OUTLIER_K,
            original_dtype=original_dtype,
        )

    # hadamard / inner_outlier_extract_left (same path per AdaHOP) /
    # outer_hadamard / none
    if mode in ("hadamard", "inner_outlier_extract_left"):
        from alto._adahop_bridge import iht_quantization
        assert hadamard_transform is not None
        g_mxfp4, g_scale = iht_quantization(grad_output, left_mul=True, use_sr=use_sr_grad)
    elif mode == "outer_hadamard":
        assert hadamard_transform is not None
        g_transformed = hadamard_transform(grad_output)
        g_mxfp4, g_scale = torch.ops.torchtitan.convert_to_mxfp4(
            g_transformed, axis=0, use_sr=use_sr_grad, is_2d_block=False,
        )
    else:  # "none"
        g_mxfp4, g_scale = torch.ops.torchtitan.convert_to_mxfp4(
            grad_output, axis=0, use_sr=use_sr_grad, is_2d_block=False,
        )

    grad_weights = _blockwise_mxfp4_gemm_or_dequant(
        g_mxfp4, g_scale, x, x_scale,
        trans_a=True, output_dtype=original_dtype,
    )
    if mode == "outer_hadamard":
        grad_weights = hadamard_transform(hadamard_transform(grad_weights, left_mul=True))
    return grad_weights
