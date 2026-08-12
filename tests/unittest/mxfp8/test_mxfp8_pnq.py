# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""PNQ invariants for MXFP8 flash attention.

The pre-PNQ implementation accumulated the online-softmax denominator from
full-precision P while its numerator used MXFP8 P.  ``V = 1`` makes this
inconsistency directly observable: a normalized attention result must be 1.
"""

import pytest
import torch

from .utils import mxfp8_attention_forward_reference


def _make_qkv(v_mode: str, device: str, dtype: torch.dtype):
    torch.manual_seed(20260812)
    shape = (1, 4, 96, 64)
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    if v_mode == "ones":
        v = torch.ones(shape, device=device, dtype=dtype)
    elif v_mode == "biased":
        v = torch.randn(shape, device=device, dtype=dtype) + 2.0
    else:
        raise ValueError(f"unsupported V mode: {v_mode}")
    return q, k, v


def _relative_l2(reference: torch.Tensor, actual: torch.Tensor) -> float:
    return (reference.float() - actual.float()).norm().div(reference.float().norm()).item()


def test_pnq_reference_preserves_constant_value_attention():
    """Quantized P must be the shared numerator and denominator distribution."""
    q, k, v = _make_qkv("ones", "cpu", torch.float32)
    sm_scale = q.shape[-1] ** -0.5

    o_pnq, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal=True)
    o_pre_pnq, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal=True, pnq=False)

    assert torch.allclose(o_pnq, torch.ones_like(o_pnq), atol=1e-5, rtol=0)
    assert (o_pre_pnq - 1).abs().max() > 1e-3


def test_pnq_reduces_biased_value_error():
    """The normalization correction must improve error on non-zero-mean values."""
    q, k, v = _make_qkv("biased", "cpu", torch.float32)
    sm_scale = q.shape[-1] ** -0.5
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale)
    o_pnq, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal=True)
    o_pre_pnq, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal=True, pnq=False)

    assert _relative_l2(o_sdpa, o_pnq) < _relative_l2(o_sdpa, o_pre_pnq)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required.")
def test_pnq_kernel_preserves_constant_value_attention():
    """The production kernel obeys the same V=1 invariant as the golden reference."""
    from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8

    q, k, v = _make_qkv("ones", "cuda", torch.bfloat16)
    output = triton_attention_mxfp8(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=q.shape[-1] ** -0.5,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=q.shape[-2],
        max_seqlens_k=k.shape[-2],
        causal=True,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]

    assert torch.allclose(output, torch.ones_like(output), atol=1e-2, rtol=0)
