# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import modeloptimizer.kernels.mxfp8.mxfp8_linear  # noqa: F401
from modeloptimizer.kernels.mxfp8.mxfp8_linear import blockwise_mxfp8_gemm_debug
from modeloptimizer.kernels.mxfp8.mxfp8_quantization import (
    BLOCK_SIZE_DEFAULT,
    is_cdna4,
)
from .utils import prepare_data


@pytest.mark.parametrize("shape", [(32, 32, 32), (256, 384, 128), (512, 1024, 256)])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("mxfp_format", ["e4m3", "e5m2"])
@pytest.mark.parametrize("data_type", [torch.bfloat16, torch.float32])
def test_mxfp8_linear_kernel(
    shape,
    trans_a,
    trans_b,
    contiguous,
    mxfp_format,
    data_type,
):
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = shape

    if trans_a:
        if contiguous:
            a = prepare_data((k, m), data_type)
        else:
            a = prepare_data((m, k), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -2
    else:
        if contiguous:
            a = prepare_data((m, k), data_type)
        else:
            a = prepare_data((k, m), data_type).transpose(-1, -2)
            assert not a.is_contiguous()
        a_axis = -1

    if trans_b:
        if contiguous:
            b = prepare_data((n, k), data_type)
        else:
            b = prepare_data((k, n), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -1
    else:
        if contiguous:
            b = prepare_data((k, n), data_type)
        else:
            b = prepare_data((n, k), data_type).transpose(-1, -2)
            assert not b.is_contiguous()
        b_axis = -2

    a_lp, a_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        a,
        mxfp_format=mxfp_format,
        axis=a_axis,
        is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        b,
        mxfp_format=mxfp_format,
        axis=b_axis,
        is_2d_block=False,
    )

    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp,
        a_scales,
        output_dtype=data_type,
        axis=a_axis,
        is_2d_block=False,
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp,
        b_scales,
        output_dtype=data_type,
        axis=b_axis,
        is_2d_block=False,
    )

    c = torch.ops.modeloptimizer.blockwise_mxfp8_gemm(
        a_lp,
        a_scales,
        b_lp,
        b_scales,
        trans_a=trans_a,
        trans_b=trans_b,
        output_dtype=data_type,
    )
    c_ref = (a_dq.T if trans_a else a_dq) @ (b_dq.T if trans_b else b_dq)

    assert torch.allclose(c_ref, c)


# =============================================================================
# DEBUG TEST CASES
# =============================================================================

@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
def test_mxfp8_linear_debug_minimal(trans_a, trans_b):
    """
    Minimal debug test for MXFP8 GEMM kernel.
    Uses smallest possible matrices (32x32x32) with a single quant block.
    Prints detailed information to help debug misalignment issues.
    """
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = 32, 32, 32
    block_size = BLOCK_SIZE_DEFAULT
    data_type = torch.float32
    mxfp_format = "e4m3"

    print(f"\n{'#'*70}")
    print(f"# TEST: trans_a={trans_a}, trans_b={trans_b}")
    print(f"# Shape: M={m}, N={n}, K={k}, block_size={block_size}")
    print(f"{'#'*70}")

    torch.manual_seed(42)

    if trans_a:
        a_hp = torch.randn((k, m), dtype=data_type, device="cuda") * 0.1
        a_axis = -2
    else:
        a_hp = torch.randn((m, k), dtype=data_type, device="cuda") * 0.1
        a_axis = -1

    if trans_b:
        b_hp = torch.randn((n, k), dtype=data_type, device="cuda") * 0.1
        b_axis = -1
    else:
        b_hp = torch.randn((k, n), dtype=data_type, device="cuda") * 0.1
        b_axis = -2

    print(f"\n--- Input tensors ---")
    print(f"a_hp.shape={a_hp.shape}, a_axis={a_axis}")
    print(f"b_hp.shape={b_hp.shape}, b_axis={b_axis}")

    a_lp, a_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        a_hp, mxfp_format=mxfp_format, axis=a_axis, is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        b_hp, mxfp_format=mxfp_format, axis=b_axis, is_2d_block=False,
    )

    print(f"\n--- Quantized tensors ---")
    print(f"a_lp.shape={a_lp.shape}, a_scales.shape={a_scales.shape}")
    print(f"b_lp.shape={b_lp.shape}, b_scales.shape={b_scales.shape}")

    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp, a_scales, output_dtype=data_type, axis=a_axis, is_2d_block=False,
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp, b_scales, output_dtype=data_type, axis=b_axis, is_2d_block=False,
    )

    a_for_matmul = a_dq.T if trans_a else a_dq
    b_for_matmul = b_dq.T if trans_b else b_dq
    c_ref = a_for_matmul @ b_for_matmul

    print(f"\n--- Running debug kernel ---")
    c = blockwise_mxfp8_gemm_debug(
        a_lp, a_scales, b_lp, b_scales,
        trans_a=trans_a, trans_b=trans_b,
        block_size=block_size, output_dtype=data_type,
        debug_print=True,
    )
    torch.cuda.synchronize()

    diff = (c - c_ref).abs()
    print(f"\n--- Comparison ---")
    print(f"max diff: {diff.max().item()}")
    print(f"c[:4,:4]:\n{c[:4, :4]}")
    print(f"c_ref[:4,:4]:\n{c_ref[:4, :4]}")

    assert torch.allclose(c_ref, c, atol=1e-3, rtol=1e-3), f"max_diff={diff.max().item()}"


@pytest.mark.parametrize("contiguous", [True, False])
def test_mxfp8_linear_debug_contiguous(contiguous):
    """Debug test to check if non-contiguous tensors cause issues."""
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = 32, 32, 32
    block_size = BLOCK_SIZE_DEFAULT
    data_type = torch.float32
    mxfp_format = "e4m3"

    print(f"\n{'#'*70}")
    print(f"# TEST: contiguous={contiguous}")
    print(f"{'#'*70}")

    torch.manual_seed(42)

    if contiguous:
        a_hp = torch.randn((m, k), dtype=data_type, device="cuda") * 0.1
        b_hp = torch.randn((k, n), dtype=data_type, device="cuda") * 0.1
    else:
        a_hp = torch.randn((k, m), dtype=data_type, device="cuda").T * 0.1
        b_hp = torch.randn((n, k), dtype=data_type, device="cuda").T * 0.1
        assert not a_hp.is_contiguous()
        assert not b_hp.is_contiguous()

    print(f"a_hp: shape={a_hp.shape}, contiguous={a_hp.is_contiguous()}, stride={a_hp.stride()}")
    print(f"b_hp: shape={b_hp.shape}, contiguous={b_hp.is_contiguous()}, stride={b_hp.stride()}")

    a_lp, a_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        a_hp, mxfp_format=mxfp_format, axis=-1, is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        b_hp, mxfp_format=mxfp_format, axis=-2, is_2d_block=False,
    )

    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp, a_scales, output_dtype=data_type, axis=-1, is_2d_block=False,
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp, b_scales, output_dtype=data_type, axis=-2, is_2d_block=False,
    )

    c_ref = a_dq @ b_dq

    c = blockwise_mxfp8_gemm_debug(
        a_lp, a_scales, b_lp, b_scales,
        trans_a=False, trans_b=False,
        block_size=block_size, output_dtype=data_type,
        debug_print=True,
    )
    torch.cuda.synchronize()

    diff = (c - c_ref).abs()
    print(f"\nmax diff: {diff.max().item()}")

    assert torch.allclose(c_ref, c, atol=1e-3, rtol=1e-3), f"max_diff={diff.max().item()}"


def test_mxfp8_linear_debug_scale_layout():
    """Debug test to verify scale tensor layouts are correct."""
    if not is_cdna4():
        pytest.skip("MXFP8 linear kernel is only available on CDNA4.")

    m, n, k = 32, 32, 64  # K=64 means 2 scale blocks
    block_size = BLOCK_SIZE_DEFAULT
    data_type = torch.float32
    mxfp_format = "e4m3"

    print(f"\n{'#'*70}")
    print(f"# TEST: Scale layout verification")
    print(f"# K={k}, block_size={block_size}, num_scale_blocks={k//block_size}")
    print(f"{'#'*70}")

    torch.manual_seed(42)

    a_hp = torch.randn((m, k), dtype=data_type, device="cuda") * 0.1
    b_hp = torch.randn((k, n), dtype=data_type, device="cuda") * 0.1

    a_lp, a_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        a_hp, mxfp_format=mxfp_format, axis=-1, is_2d_block=False,
    )
    b_lp, b_scales = torch.ops.modeloptimizer.convert_to_mxfp8(
        b_hp, mxfp_format=mxfp_format, axis=-2, is_2d_block=False,
    )

    print(f"a_lp.shape={a_lp.shape}, a_scales.shape={a_scales.shape}")
    print(f"Expected a_scales.shape: [{m}, {k//block_size}]")
    print(f"b_lp.shape={b_lp.shape}, b_scales.shape={b_scales.shape}")
    print(f"Expected b_scales.shape: [{k//block_size}, {n}]")

    assert a_scales.shape == (m, k // block_size)
    assert b_scales.shape == (k // block_size, n)

    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp, a_scales, output_dtype=data_type, axis=-1, is_2d_block=False,
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp, b_scales, output_dtype=data_type, axis=-2, is_2d_block=False,
    )

    c_ref = a_dq @ b_dq

    print(f"\n--- Kernel stride handling ---")
    print(f"b_scales shape: {b_scales.shape}, stride: {b_scales.stride()}")

    c = blockwise_mxfp8_gemm_debug(
        a_lp, a_scales, b_lp, b_scales,
        trans_a=False, trans_b=False,
        block_size=block_size, output_dtype=data_type,
        debug_print=True,
    )
    torch.cuda.synchronize()

    diff = (c - c_ref).abs()
    print(f"\nmax diff: {diff.max().item()}")

    assert torch.allclose(c_ref, c, atol=1e-3, rtol=1e-3), f"max_diff={diff.max().item()}"
