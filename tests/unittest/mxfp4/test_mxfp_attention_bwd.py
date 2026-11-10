# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch

from alto.kernels.fp4.mxfp4.mxfp_quantization import convert_from_mxfp4, convert_to_mxfp4
from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import (
    _triton_attention_mxfp4,
    triton_attention_mxfp4,
)


def _qdq(x, is_2d_block):
    packed, scale = convert_to_mxfp4(x.float(), axis=-1, is_2d_block=is_2d_block)
    return convert_from_mxfp4(packed, scale, axis=-1, is_2d_block=is_2d_block, output_dtype=torch.float32)


def _online_reference(q, k, v, sm_scale, causal, hadamard, block=64):
    """Online softmax whose row sum and PV both use the dequantized numerator."""
    qh = _qdq(hadamard(q), True)
    kh = _qdq(hadamard(k), True)
    vh = _qdq(v, True)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * sm_scale
    if causal:
        index = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores.masked_fill(index[:, None] < index[None, :], float("-inf"))
    seqlen = scores.shape[-1]
    pad = (block - seqlen % block) % block
    tiled = torch.nn.functional.pad(scores, (0, pad), value=float("-inf")) if pad else scores
    value = torch.nn.functional.pad(vh, (0, 0, 0, pad)) if pad else vh
    running_max = torch.full(scores.shape[:-1], float("-inf"), device=scores.device)
    normalizer = torch.ones(scores.shape[:-1], device=scores.device)
    acc = torch.zeros(*scores.shape[:-1], vh.shape[-1], device=scores.device)
    for start in range(0, tiled.shape[-1], block):
        tile = tiled[..., start:start + block]
        tile_max = torch.where(torch.isfinite(tile), tile, float("-inf")).amax(dim=-1)
        m_ij = torch.maximum(running_max, tile_max)
        diff = running_max - m_ij
        alpha = torch.exp(diff)
        alpha = torch.where(torch.isfinite(diff), alpha, torch.where(torch.isfinite(m_ij), torch.zeros_like(alpha), torch.ones_like(alpha)))
        shifted = torch.where(torch.isfinite(tile), torch.exp(tile - m_ij.unsqueeze(-1)), torch.zeros_like(tile))
        p_deq = _qdq(shifted, False)
        normalizer = normalizer * alpha + p_deq.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.matmul(p_deq, value[:, :, start:start + block, :])
        running_max = m_ij
    output = acc / normalizer.unsqueeze(-1)
    return output, scores, qh, kh, vh


def _cosine(actual, expected):
    return torch.nn.functional.cosine_similarity(actual.float().reshape(-1), expected.reshape(-1), dim=0)


def test_constant_values_stay_normalized():
    seqlen, heads, dim = 128, 2, 128
    q = torch.randn(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    v = torch.ones(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
    output = triton_attention_mxfp4(
        q, k, v, None, None, dim**-0.5, 0.0, 0, 0, seqlen, seqlen, True, False, False, "bhsd"
    )[0]
    assert torch.allclose(output.float(), torch.ones_like(output.float()))


def test_backward_matches_dequantized_probability():
    torch.manual_seed(0)
    batch, seqlen, heads, dim = 2, 96, 2, 128
    q = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    v = torch.randn(batch, seqlen, heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    scale = dim**-0.5
    output, lse, _ = triton_attention_mxfp4(
        q, k, v, None, None, scale, 0.0, 0, 0, seqlen, seqlen, True, False, False, "bhsd"
    )
    hadamard = _triton_attention_mxfp4.last_qk_hadamard
    # A later forward builds a new transform. Backward must keep the one saved above.
    triton_attention_mxfp4(
        q.detach(), k.detach(), v.detach(), None, None, scale, 0.0, 0, 0, seqlen, seqlen, True, False, False, "bhsd"
    )
    reference, scores, qh, kh, vh = _online_reference(
        q.detach(), k.detach(), v.detach(), scale, True, hadamard)
    assert (output.float() - reference).abs().max() < 2e-2

    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    finite = torch.isfinite(scores)
    probability = torch.where(finite, torch.exp(scores - lse.detach().unsqueeze(-1)), torch.zeros_like(scores))
    grad_deq = _qdq(grad_output.float(), False)
    d_probability = torch.matmul(grad_deq, vh.transpose(-1, -2))
    delta = (output.float() * grad_deq).sum(dim=-1, keepdim=True)
    d_scores = probability * (d_probability - delta) * scale
    expected = (
        hadamard(torch.matmul(d_scores, kh), inverse=True),
        hadamard(torch.matmul(d_scores.transpose(-1, -2), qh), inverse=True),
        torch.matmul(probability.transpose(-1, -2), grad_deq),
    )
    for actual, target in zip((q.grad, k.grad, v.grad), expected):
        assert torch.isfinite(actual).all()
        assert _cosine(actual, target) > 0.999
