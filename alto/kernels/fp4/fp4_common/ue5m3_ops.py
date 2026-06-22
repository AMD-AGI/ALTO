# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""UE5M3 (unsigned 5-exp / 3-mant, NaN-aware) dtype primitives.

UE5M3 is the inner-block scale format used by the AMD-FP4 recipe (NVFP4
1x16 micro-block layout + UE5M3 inner scale + FP32 per-tensor outer
scale).  It trades the sign bit of E4M3 (which is unused for scales,
which are always >= 0) for one extra exponent bit, **doubling the
exponent range at zero precision cost** (mantissa width stays at 3).

Specification (committed, GFXIPARCH-2067 §19.10 OCP E5M3 aligned):

* bit layout: ``[7:3] = exponent (5b)``, ``[2:0] = mantissa (3b)``;
  no sign bit.
* exponent bias: ``2^(5-1) - 1 = 15``.
* code 0x00          -> +0.0
* codes 0x01..0x07   -> subnormals: ``value = (mant/8) * 2^-14``
* codes 0x08..0xFE   -> normals:    ``value = (1 + mant/8) * 2^(exp-15)``;
  ``0xFE -> (1 + 6/8) * 2^16 = 114688.0`` (UE5M3_MAX, max normal).
* code 0xFF          -> NaN (reserved).
* Inf encoding: **none**.  ±Inf, NaN, and any value strictly greater
  than ``UE5M3_MAX`` encode to ``0xFF`` (NaN code).  Finite negative
  inputs clamp to 0 (UE5M3 is unsigned).

Decision history:

* D2 (original, finite-only): ``0xFF -> 122880`` saturate-on-overflow.
  Superseded; broke bit-level alignment with GFXIPARCH-2067 §19.10.
* D2' (current): ``0xFE`` is max normal, ``0xFF`` is NaN; NaN
  propagation into downstream GEMM scales is prevented by a defense
  layer in :mod:`alto.kernels.fp4.nvfp4.nvfp_quantization` and
  :mod:`tests.unittest.nvfp4.utils._quantize_inner_scale` (mirrors
  TransformerEngine / vLLM / TRT-LLM "caller-side sanitize" pattern).

This module exposes two surfaces:

* PyTorch reference (this is the bit-level oracle):
  :func:`f32_to_ue5m3_uint8`, :func:`ue5m3_uint8_to_f32`,
  :func:`quantize_to_ue5m3` (round-trip convenience).
* Triton ``@triton.jit`` factories used by the NVFP4 quantization
  kernels: :func:`make_quantize_ue5m3` (returns an in-kernel
  function that snaps an fp32 to the UE5M3 grid).

The PyTorch and Triton paths are required to be bit-identical on the
fp32 snapped value across the full 256-code grid; see
``tests/unittest/ue5m3/`` for the oracle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UE5M3_NUM_EXP_BITS = 5
UE5M3_NUM_MAN_BITS = 3
UE5M3_EXP_BIAS = (1 << (UE5M3_NUM_EXP_BITS - 1)) - 1  # 15

# Max normal (code 0xFE per GFXIPARCH-2067 §19.10):
#   (1 + 6/8) * 2^(31-15) = 1.75 * 65536 = 114688.0
UE5M3_MAX = 1.75 * (2.0 ** (((1 << UE5M3_NUM_EXP_BITS) - 1) - UE5M3_EXP_BIAS))

# Smallest positive normal (code 0x08): 2^(1-15) = 2^-14
UE5M3_MIN_NORMAL = 2.0 ** (1 - UE5M3_EXP_BIAS)

# Smallest positive subnormal (code 0x01): (1/8) * 2^-14 = 2^-17
UE5M3_EPS = (1.0 / (1 << UE5M3_NUM_MAN_BITS)) * UE5M3_MIN_NORMAL

# Reserved NaN code (per GFXIPARCH-2067 §19.10).
UE5M3_NAN_CODE = 0xFF
# Quiet-NaN bit pattern used by the decoder when emitting fp32 NaN.
_FP32_QUIET_NAN_BITS = 0x7FC00000

# Sanity: the chosen FP32 div-by-zero floor used by NVFP4 outer-scale code
# must keep ``outer_scale * UE5M3_EPS`` inside FP32 normal range so that the
# effective per-block divisor never flushes under FTZ.  We re-verify here
# at import time so any future floor change breaks loudly.
_FP32_MIN_NORMAL = torch.finfo(torch.float32).tiny
_OUTER_SCALE_DIVZERO_FLOOR = 1.0e-30  # mirrors nvfp_quantization._OUTER_SCALE_DIVZERO_FLOOR
assert _OUTER_SCALE_DIVZERO_FLOOR * UE5M3_EPS > _FP32_MIN_NORMAL, (
    "UE5M3_EPS * _OUTER_SCALE_DIVZERO_FLOOR falls below FP32 min normal; "
    "the NVFP4 outer-scale floor in nvfp_quantization.py must be raised."
)
del _FP32_MIN_NORMAL


# Internal float32 IEEE-754 layout constants
_F32_EXP_BITS = 8
_F32_MAN_BITS = 23
_F32_EXP_BIAS = (1 << (_F32_EXP_BITS - 1)) - 1  # 127


# ---------------------------------------------------------------------------
# PyTorch reference -- bit-level oracle.
# Inputs/outputs match the Triton path bit-for-bit on the fp32 snapped value.
# ---------------------------------------------------------------------------

def f32_to_ue5m3_uint8(x: torch.Tensor) -> torch.Tensor:
    """Encode a non-negative FP32 tensor to packed UE5M3 (uint8).

    Behavior (D2', GFXIPARCH-2067 §19.10):
        * round-to-nearest-even (ties to even via magic-adder trick)
        * ``UE5M3_MAX`` (114688) encodes to ``0xFE`` (max normal)
        * NaN, ±Inf, and finite values strictly greater than ``UE5M3_MAX``
          encode to ``0xFF`` (NaN code, per spec)
        * finite negative inputs clamp to 0 (unsigned format); -Inf
          counts as a special and encodes to ``0xFF``

    Args:
        x: float32 tensor (CPU or CUDA).

    Returns:
        uint8 tensor of identical shape; each element holds a UE5M3 code.
    """
    assert x.dtype == torch.float32, f"f32_to_ue5m3_uint8 expects float32, got {x.dtype}"

    # ---- Identify "special" inputs that must encode to the reserved NaN code
    # 0xFF: NaN, ±Inf, and finite values that overflow the normal range.
    # We short-circuit them to 0 so the RTNE bit-magic only sees well-formed
    # non-negative finite values in [0, UE5M3_MAX], then re-tag them at the end.
    is_nan_or_inf = torch.isnan(x) | torch.isinf(x)
    is_overflow = x > UE5M3_MAX
    is_special = is_nan_or_inf | is_overflow

    x = torch.where(is_special, torch.zeros_like(x), x)
    x = torch.clamp(x, min=0.0, max=UE5M3_MAX)

    # Saturate mask AFTER clamp: items pinned exactly at UE5M3_MAX (= 0xFE)
    # must encode to the max-normal code, NOT to the NaN code.
    saturate_mask = x >= UE5M3_MAX
    denormal_mask = (~saturate_mask) & (x < UE5M3_MIN_NORMAL)
    normal_mask = ~(saturate_mask | denormal_mask)

    # ---- Normal path: magic-adder RTNE on the lower (23-3) = 20 bits.
    #
    #     val_to_add = ((EXP_BIAS_LP - EXP_BIAS_F32) << 23)   <-- exponent rebias
    #                  + magic_adder                          <-- round-to-nearest
    #                  + mant_odd                             <-- tie-to-even
    #     out = ((x_int + val_to_add) >> (23 - 3))            <-- truncate
    #
    # This is the standard textbook trick (see Computer Organization and
    # Design, RISC-V edition, Chapter 3.5) and mirrors what
    # ``alto.kernels.fp4.testing_utils._f32_to_floatx_unpacked`` does for
    # the signed FloatX path, minus the sign-bit handling.
    magic_adder = (1 << (_F32_MAN_BITS - UE5M3_NUM_MAN_BITS - 1)) - 1  # 524287

    x_int = x.view(torch.int32)
    mant_odd = (x_int >> (_F32_MAN_BITS - UE5M3_NUM_MAN_BITS)) & 1
    val_to_add = ((UE5M3_EXP_BIAS - _F32_EXP_BIAS) << _F32_MAN_BITS) + magic_adder
    normal_x = x_int + val_to_add + mant_odd
    normal_x = (normal_x >> (_F32_MAN_BITS - UE5M3_NUM_MAN_BITS)).to(torch.uint8)

    # ---- Denormal path: align-by-add trick.
    #
    # Adding ``denorm_mask_float`` shifts the input so that the bits
    # representing the subnormal mantissa land exactly where UE5M3 wants
    # them; subtracting the integer of the same value isolates those bits.
    denorm_exp = (_F32_EXP_BIAS - UE5M3_EXP_BIAS) + (_F32_MAN_BITS - UE5M3_NUM_MAN_BITS) + 1
    denorm_mask_int = denorm_exp << _F32_MAN_BITS
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32).item()

    denormal_x = (x + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # ---- Compose: max-normal 0xFE is the default for the saturate path;
    # overwrite for normal/denormal paths; finally re-tag special inputs to
    # the NaN code 0xFF (D2', spec-aligned).
    out = torch.full_like(x, 0xFE, dtype=torch.uint8)
    out = torch.where(normal_mask, normal_x, out)
    out = torch.where(denormal_mask, denormal_x, out)
    nan_code = torch.full_like(out, UE5M3_NAN_CODE)
    out = torch.where(is_special, nan_code, out)
    return out


def ue5m3_uint8_to_f32(qx: torch.Tensor) -> torch.Tensor:
    """Decode a UE5M3 uint8 tensor to FP32.

    This is the canonical decoder; the value table is (D2', spec-aligned):

        code 0x00          -> 0.0
        codes 0x01..0x07   -> (mant/8) * 2^-14
        codes 0x08..0xFE   -> (1 + mant/8) * 2^(exp-15)
        code 0xFF          -> NaN (reserved per GFXIPARCH-2067 §19.10)

    Args:
        qx: uint8 tensor.

    Returns:
        float32 tensor of identical shape.
    """
    assert qx.dtype == torch.uint8, f"ue5m3_uint8_to_f32 expects uint8, got {qx.dtype}"

    exp = (qx >> UE5M3_NUM_MAN_BITS).to(torch.int32)            # [0, 31]
    mant = (qx & ((1 << UE5M3_NUM_MAN_BITS) - 1)).to(torch.int32)

    nan_code_mask = qx == UE5M3_NAN_CODE
    zero_mask = qx == 0
    denormal_mask = (exp == 0) & (mant != 0)
    normal_mask = (exp > 0) & ~nan_code_mask

    # Normal path: rebias exponent into FP32, shift mantissa.
    exp_f32 = (exp - UE5M3_EXP_BIAS + _F32_EXP_BIAS) << _F32_MAN_BITS
    mant_f32 = mant << (_F32_MAN_BITS - UE5M3_NUM_MAN_BITS)
    normal_int = exp_f32 | mant_f32

    # Denormal path: with 3-bit mantissa we enumerate the seven non-zero codes
    # directly; this avoids the loop-based approach of the signed FloatX helper
    # for clarity at this small size.
    # value(mant) = (mant / 8) * 2^-14
    #             = mant * 2^-17
    # In FP32 bits, that's:
    #   mant == 1: 2^-17                = (127 - 17) << 23                  = 0x37000000
    #   mant == 2: 2^-16                = (127 - 16) << 23                  = 0x37800000
    #   mant == 3: 1.5 * 2^-16          = ((127-16) << 23) | (1 << 22)      = 0x37C00000
    #   mant == 4: 2^-15                = (127 - 15) << 23                  = 0x38000000
    #   mant == 5: 1.25 * 2^-15         = ((127-15) << 23) | (1 << 21)      = 0x38200000
    #   mant == 6: 1.5  * 2^-15         = ((127-15) << 23) | (2 << 21)      = 0x38400000
    #   mant == 7: 1.75 * 2^-15         = ((127-15) << 23) | (3 << 21)      = 0x38600000
    denorm_table = torch.tensor(
        [0x00000000,  # mant=0 (unreachable on denormal path; placeholder)
         0x37000000, 0x37800000, 0x37C00000,
         0x38000000, 0x38200000, 0x38400000, 0x38600000],
        dtype=torch.int32,
        device=qx.device,
    )
    denormal_int = denorm_table[mant]

    result = torch.where(normal_mask, normal_int, torch.zeros_like(normal_int))
    result = torch.where(denormal_mask, denormal_int, result)
    result = torch.where(zero_mask, torch.zeros_like(result), result)
    # Spec-aligned NaN exit: code 0xFF decodes to a quiet FP32 NaN.
    result = torch.where(
        nan_code_mask,
        torch.full_like(result, _FP32_QUIET_NAN_BITS),
        result,
    )

    return result.view(torch.float32)


def quantize_to_ue5m3(x: torch.Tensor) -> torch.Tensor:
    """Round an FP32 tensor to the UE5M3 grid, returning FP32.

    Equivalent to ``ue5m3_uint8_to_f32(f32_to_ue5m3_uint8(x))`` and
    therefore idempotent: ``quantize_to_ue5m3(quantize_to_ue5m3(x)) ==
    quantize_to_ue5m3(x)`` bit-for-bit.

    This is the surface used by the NVFP4 quant kernel's inner-scale
    rounding step (the AMD-FP4 analogue of ``inner.to(torch.float8_e4m3fn)
    .to(torch.float32)``).
    """
    return ue5m3_uint8_to_f32(f32_to_ue5m3_uint8(x))


# ---------------------------------------------------------------------------
# Triton ``@triton.jit`` factories.
#
# The factory pattern mirrors :mod:`alto.kernels.fp4.fp4_primitives.triton_fp4_ops`
# (``make_quantize_e2m1``) so call-sites can swap one for another without
# changing the surrounding kernel skeleton.
#
# The returned function takes a positive FP32 tile and returns the
# UE5M3-snapped FP32 tile.  No uint8 surface is exposed inside Triton
# because the only consumer (``_calculate_nvfp4_scales``) needs the
# snapped fp32 value as a divisor.
# ---------------------------------------------------------------------------

def _ue5m3_triton_constants() -> dict:
    """Pre-compute all Triton-side magic numbers in pure Python.

    Triton's ``tl.constexpr`` slots accept Python ints / floats but not the
    result of in-kernel ops (``tl.cast`` etc.), so we resolve the FP32
    align-by-add value here and pass it in as a float constant.
    """
    ebits = UE5M3_NUM_EXP_BITS
    mbits = UE5M3_NUM_MAN_BITS
    exp_bias_lp = UE5M3_EXP_BIAS
    f32_man_bits = _F32_MAN_BITS
    f32_exp_bias = _F32_EXP_BIAS

    denorm_exp = (f32_exp_bias - exp_bias_lp) + (f32_man_bits - mbits) + 1   # 133
    denorm_mask_int = denorm_exp << f32_man_bits                             # 0x42800000
    # Bitcast 0x42800000 -> fp32: sign=0, exp=133, mant=0 -> 1.0 * 2^(133-127) = 64.0
    denorm_mask_float = float(
        torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32).item()
    )

    # Signed val_to_add for the normal-path rebias + RTNE adder; this is an
    # int32 quantity that wraps correctly when added to a uint32 bit pattern.
    magic_adder = (1 << (f32_man_bits - mbits - 1)) - 1                      # 524287
    val_to_add = ((exp_bias_lp - f32_exp_bias) << f32_man_bits) + magic_adder

    return {
        "EBITS": ebits,
        "MBITS": mbits,
        "EXP_BIAS_LP": exp_bias_lp,
        "MBITS_F32": f32_man_bits,
        "EXP_BIAS_F32": f32_exp_bias,
        "MAX_VAL": UE5M3_MAX,
        "MIN_NORMAL": UE5M3_MIN_NORMAL,
        "MAGIC_ADDER": magic_adder,
        "VAL_TO_ADD": val_to_add,
        "DENORM_MASK_INT": denorm_mask_int,
        "DENORM_MASK_FLOAT": denorm_mask_float,
        "NAN_CODE": UE5M3_NAN_CODE,
        "NAN_BITS": _FP32_QUIET_NAN_BITS,
    }


# Resolved once at import time -- these values never change.
_UE5M3_CONSTS = _ue5m3_triton_constants()
# Sanity: 64.0 is the only correct value for the align-by-add float.
assert _UE5M3_CONSTS["DENORM_MASK_FLOAT"] == 64.0, _UE5M3_CONSTS


def make_quantize_ue5m3():
    """Return a ``@triton.jit`` function that snaps FP32 to the UE5M3 grid.

    Returned signature::

        _quantize_ue5m3(x: tl.tensor[fp32]) -> tl.tensor[fp32]

    Semantics are bit-identical to :func:`quantize_to_ue5m3`.
    """

    # Capture as Python locals so each tl.constexpr below can read a plain int / float.
    _MBITS_PY = _UE5M3_CONSTS["MBITS"]
    _EXP_BIAS_LP_PY = _UE5M3_CONSTS["EXP_BIAS_LP"]
    _MBITS_F32_PY = _UE5M3_CONSTS["MBITS_F32"]
    _EXP_BIAS_F32_PY = _UE5M3_CONSTS["EXP_BIAS_F32"]
    _MAX_VAL_PY = _UE5M3_CONSTS["MAX_VAL"]
    _MIN_NORMAL_PY = _UE5M3_CONSTS["MIN_NORMAL"]
    _VAL_TO_ADD_PY = _UE5M3_CONSTS["VAL_TO_ADD"]
    _DENORM_MASK_INT_PY = _UE5M3_CONSTS["DENORM_MASK_INT"]
    _DENORM_MASK_FLOAT_PY = _UE5M3_CONSTS["DENORM_MASK_FLOAT"]
    _NAN_CODE_PY = _UE5M3_CONSTS["NAN_CODE"]
    _NAN_BITS_PY = _UE5M3_CONSTS["NAN_BITS"]

    @triton.jit
    def _quantize_ue5m3(x):
        MBITS: tl.constexpr = _MBITS_PY
        EXP_BIAS_LP: tl.constexpr = _EXP_BIAS_LP_PY
        MBITS_F32: tl.constexpr = _MBITS_F32_PY
        EXP_BIAS_F32: tl.constexpr = _EXP_BIAS_F32_PY
        MAX_VAL: tl.constexpr = _MAX_VAL_PY
        MIN_NORMAL: tl.constexpr = _MIN_NORMAL_PY
        VAL_TO_ADD: tl.constexpr = _VAL_TO_ADD_PY
        DENORM_MASK_INT: tl.constexpr = _DENORM_MASK_INT_PY
        DENORM_MASK_FLOAT: tl.constexpr = _DENORM_MASK_FLOAT_PY
        NAN_CODE: tl.constexpr = _NAN_CODE_PY
        NAN_BITS: tl.constexpr = _NAN_BITS_PY

        # ---- Identify "special" inputs that must encode to the reserved
        # NaN code 0xFF: NaN, ±Inf, and finite values strictly above MAX.
        #   * ``x != x``        catches NaN
        #   * ``x * 0 != 0``    catches NaN OR ±Inf  (Inf*0 == NaN, finite*0 == 0)
        #   * ``x > MAX_VAL``   catches finite overflow
        is_special = (x != x) | (x * 0.0 != 0.0) | (x > MAX_VAL)

        # Short-circuit specials to 0 so the RTNE bit-magic only sees
        # well-formed non-negative finite values in [0, MAX_VAL].  Negative
        # finite values clamp to 0 (unsigned).
        x = tl.where(is_special, 0.0, x)
        x = tl.minimum(x, MAX_VAL)
        x = tl.maximum(x, 0.0)

        x_int = x.to(tl.uint32, bitcast=True)
        saturate_mask = x >= MAX_VAL
        denormal_mask = (~saturate_mask) & (x < MIN_NORMAL)
        normal_mask = ~(saturate_mask | denormal_mask)

        # ---- Normal path: magic-adder RTNE
        # Use signed int32 bit patterns for the add (HIP rejects uint32 + negative).
        mant_odd = (x_int >> (MBITS_F32 - MBITS)) & 1
        normal_x = x_int.to(tl.int32, bitcast=True) + VAL_TO_ADD + mant_odd
        normal_x = (normal_x >> (MBITS_F32 - MBITS)).to(tl.uint8)

        # ---- Denormal path: align-by-add
        denormal_x = (x + DENORM_MASK_FLOAT).to(tl.uint32, bitcast=True) - DENORM_MASK_INT
        denormal_x = denormal_x.to(tl.uint8)

        # ---- Compose uint8 code (default = max-normal 0xFE for the saturate
        # path; specials get re-tagged to 0xFF below).
        # We use ``x_int * 0 + 0xFE`` instead of ``tl.full`` because we want
        # a tile-shaped result without knowing the BLOCK_SIZE here.
        encoded = (x_int * 0 + 0xFE).to(tl.uint8)
        encoded = tl.where(normal_mask, normal_x, encoded)
        encoded = tl.where(denormal_mask, denormal_x, encoded)
        # Re-tag specials to the NaN code (D2', spec-aligned).
        encoded = tl.where(is_special, NAN_CODE, encoded)

        # ---- Decode back to FP32 (mirror of ue5m3_uint8_to_f32, kept inline
        # so the call-site only ever sees fp32).
        exp = (encoded >> MBITS).to(tl.int32)
        mant = (encoded & 0x7).to(tl.int32)

        # Normal: ((exp - 15 + 127) << 23) | (mant << 20)
        normal_int = ((exp - EXP_BIAS_LP + EXP_BIAS_F32) << MBITS_F32) | (mant << (MBITS_F32 - MBITS))

        # Denormal: arithmetic FP32 reconstruction.
        # Let k = floor(log2(mant)), 2^k <= mant < 2^(k+1).  Then:
        #   value = (1 + (mant - 2^k) / 2^k) * 2^(k - 17)
        #   exp_f32_bits  = (k + 110) << 23
        #   mant_f32_bits = (mant - 2^k) << (23 - k)
        is_ge_4 = mant >= 4
        is_ge_2 = mant >= 2
        k = tl.where(is_ge_4, 2, tl.where(is_ge_2, 1, 0))
        two_k = tl.where(is_ge_4, 4, tl.where(is_ge_2, 2, 1))
        shift = tl.where(is_ge_4, 21, tl.where(is_ge_2, 22, 23))
        denorm_exp_f32 = (k + 110) << MBITS_F32
        denorm_mant_f32 = (mant - two_k) << shift
        denormal_int_out = denorm_exp_f32 | denorm_mant_f32

        nan_code_mask = encoded == NAN_CODE
        normal_mask_out = (exp > 0) & ~nan_code_mask
        denormal_mask_out = (exp == 0) & (mant != 0)
        # ``encoded == 0`` falls through to the final ``0`` default.
        result_int = tl.where(
            normal_mask_out,
            normal_int,
            tl.where(denormal_mask_out, denormal_int_out, 0),
        )
        # Spec-aligned NaN exit: bit-cast quiet NaN for code 0xFF.
        result_int = tl.where(nan_code_mask, NAN_BITS, result_int)

        return result_int.to(tl.float32, bitcast=True)

    return _quantize_ue5m3


# Module-level pre-built instance for callers that just want the JIT.
quantize_ue5m3 = make_quantize_ue5m3()


# A parent ``@triton.jit`` kernel can inline ``quantize_ue5m3`` from another
# module as long as the symbol is *imported* into the parent's module globals
# (Triton resolves nested JIT callees by name from the caller's globals, then
# inlines them).  For example ``nvfp_quantization._calculate_nvfp4_scales``
# does ``from alto.kernels.fp4.fp4_common import quantize_ue5m3`` and calls it
# directly.  What does NOT work is referencing the callee by a module-qualified
# attribute path (e.g. ``ue5m3_ops.quantize_ue5m3``).  This standalone snap
# driver lives here so the import is local and the rule is trivially satisfied.
@triton.jit
def _ue5m3_snap_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = quantize_ue5m3(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_quantize_to_ue5m3(x: torch.Tensor) -> torch.Tensor:
    """Snap FP32 to the UE5M3 grid on GPU (bit-identical to :func:`quantize_to_ue5m3`)."""
    assert x.is_cuda and x.dtype == torch.float32
    flat = x.contiguous().view(-1)
    out = torch.empty_like(flat)
    n = flat.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _ue5m3_snap_kernel[grid](flat, out, n, BLOCK_SIZE=block)
    return out.view_as(x)


__all__ = (
    "UE5M3_EPS",
    "UE5M3_EXP_BIAS",
    "UE5M3_MAX",
    "UE5M3_MIN_NORMAL",
    "UE5M3_NAN_CODE",
    "UE5M3_NUM_EXP_BITS",
    "UE5M3_NUM_MAN_BITS",
    "f32_to_ue5m3_uint8",
    "make_quantize_ue5m3",
    "quantize_to_ue5m3",
    "quantize_ue5m3",
    "triton_quantize_to_ue5m3",
    "ue5m3_uint8_to_f32",
)
