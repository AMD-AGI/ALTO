# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from alto.kernels.fp4.mxfp4.mxfp_quantization import convert_from_mxfp4, convert_to_mxfp4
from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import (
    _triton_attention_mxfp4,
    triton_attention_mxfp4,
)


def _qdq(x, is_2d_block):
    packed, scale = convert_to_mxfp4(x.float(), axis=-1, is_2d_block=is_2d_block)
    return convert_from_mxfp4(packed, scale, axis=-1, is_2d_block=is_2d_block, output_dtype=torch.float32)


def _apply_window(scores, window_left, window_right):
    if window_left < 0 and window_right < 0:
        return scores
    seqlen_q, seqlen_k = scores.shape[-2], scores.shape[-1]
    q_idx = torch.arange(seqlen_q, device=scores.device)[:, None]
    k_idx = torch.arange(seqlen_k, device=scores.device)[None, :]
    shift = seqlen_k - seqlen_q
    keep = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=scores.device)
    if window_left >= 0:
        keep = keep & (k_idx >= q_idx + shift - window_left)
    if window_right >= 0:
        keep = keep & (k_idx <= q_idx + shift + window_right)
    return scores.masked_fill(~keep, float("-inf"))


def _online_reference(q, k, v, sm_scale, causal, hadamard, block=64, window_size=(-1, -1)):
    """Online softmax whose row sum and PV both use the dequantized numerator."""
    qh = _qdq(hadamard(q), True)
    kh = _qdq(hadamard(k), True)
    vh = _qdq(v, True)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * sm_scale
    if causal:
        index = torch.arange(scores.shape[-1], device=scores.device)
        scores = scores.masked_fill(index[:, None] < index[None, :], float("-inf"))
    scores = _apply_window(scores, window_size[0], window_size[1])
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


def test_sliding_window_matches_dequantized_probability():
    torch.manual_seed(0)
    batch, seqlen, heads, dim = 1, 128, 2, 128
    for causal, window in ((True, (32, 0)), (False, (40, 16))):
        q = torch.randn(batch, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(batch, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        scale = dim**-0.5
        output, lse, _ = triton_attention_mxfp4(
            q, k, v, None, None, scale, 0.0, 0, 0, seqlen, seqlen, causal, False, False, "bhsd", window_size=window
        )
        hadamard = _triton_attention_mxfp4.last_qk_hadamard
        reference, scores, qh, kh, vh = _online_reference(
            q.detach(), k.detach(), v.detach(), scale, causal, hadamard, window_size=window)
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


def test_thd_quantizes_on_sequence_dim_plane():
    from alto.kernels.hadamard_transform.transform import HadamardFactory

    class _Identity:
        def __call__(self, value, inverse=False, left_mul=False):
            del inverse, left_mul
            return value

    raw = HadamardFactory.__dict__["create_transform"]
    HadamardFactory.create_transform = classmethod(lambda cls, device: _Identity())
    try:
        torch.manual_seed(0)
        heads, seqlen, dim = 4, 64, 64
        q_bhsd = torch.randn(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k_bhsd = torch.randn(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v_bhsd = torch.randn(1, heads, seqlen, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        q_bshd = q_bhsd.detach().transpose(1, 2).requires_grad_()
        k_bshd = k_bhsd.detach().transpose(1, 2).requires_grad_()
        v_bshd = v_bhsd.detach().transpose(1, 2).requires_grad_()
        q_thd = q_bhsd.detach()[0].transpose(0, 1).requires_grad_()
        k_thd = k_bhsd.detach()[0].transpose(0, 1).requires_grad_()
        v_thd = v_bhsd.detach()[0].transpose(0, 1).requires_grad_()
        scale = dim**-0.5
        common = dict(bias=None, alibi_slopes=None, sm_scale=scale, dropout_p=0.0, causal=True, return_scores=False,
                      use_exp2=True, window_size=(-1, -1))
        out_bhsd, lse_bhsd, _ = triton_attention_mxfp4(
            q_bhsd, k_bhsd, v_bhsd, cu_seqlens_q=0, cu_seqlens_k=0, max_seqlens_q=seqlen, max_seqlens_k=seqlen,
            layout="bhsd", **common)
        out_bshd, lse_bshd, _ = triton_attention_mxfp4(
            q_bshd, k_bshd, v_bshd, cu_seqlens_q=0, cu_seqlens_k=0, max_seqlens_q=seqlen, max_seqlens_k=seqlen,
            layout="bshd", **common)
        cu = torch.tensor([0, seqlen], device="cuda", dtype=torch.int32)
        out_thd, lse_thd, _ = triton_attention_mxfp4(
            q_thd, k_thd, v_thd, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlens_q=seqlen, max_seqlens_k=seqlen,
            layout="thd", **common)
        assert not q_bshd.is_contiguous()
        assert not q_thd.is_contiguous()
        assert torch.equal(out_bshd.transpose(1, 2), out_bhsd)
        assert torch.equal(lse_bshd, lse_bhsd)
        assert torch.equal(out_thd.transpose(0, 1), out_bhsd[0])
        assert torch.equal(lse_thd.transpose(0, 1), lse_bhsd[0])
        out_bhsd.sum().backward()
        out_bshd.sum().backward()
        out_thd.sum().backward()
        assert torch.equal(q_bshd.grad.transpose(1, 2), q_bhsd.grad)
        assert torch.equal(k_bshd.grad.transpose(1, 2), k_bhsd.grad)
        assert torch.equal(v_bshd.grad.transpose(1, 2), v_bhsd.grad)
        assert torch.equal(q_thd.grad.transpose(0, 1), q_bhsd.grad[0])
        assert torch.equal(k_thd.grad.transpose(0, 1), k_bhsd.grad[0])
        assert torch.equal(v_thd.grad.transpose(0, 1), v_bhsd.grad[0])
    finally:
        HadamardFactory.create_transform = raw


def test_varlen_int64_cu_seqlens_keeps_sequences():
    from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import attention_mxfp4_forward_triton_impl
    from alto.kernels.hadamard_transform.transform import HadamardFactory

    class _Identity:
        def __call__(self, value, inverse=False, left_mul=False):
            del inverse, left_mul
            return value

    raw = HadamardFactory.__dict__["create_transform"]
    HadamardFactory.create_transform = classmethod(lambda cls, device: _Identity())
    try:
        torch.manual_seed(0)
        heads, dim = 4, 64
        lengths = (32, 64)
        total = sum(lengths)
        q = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu64 = torch.tensor([0, lengths[0], total], device="cuda", dtype=torch.int64)
        cu32 = cu64.to(torch.int32)
        scale = dim**-0.5

        def run(cu, query, key, value):
            return triton_attention_mxfp4(
                query, key, value, None, None, scale, 0.0, cu, cu, max(lengths), max(lengths), True, False, True,
                "thd", window_size=(-1, 0))

        out64, lse64, _ = run(cu64, q, k, v)
        out32, lse32, _ = run(cu32, q, k, v)
        assert torch.equal(out64, out32)
        assert torch.equal(lse64, lse32)
        assert out64.shape == q.shape

        k_other = k.clone()
        k_other[lengths[0]:] += 1
        out_other, _, _ = run(cu64, q, k_other, v)
        assert torch.equal(out64[:lengths[0]], out_other[:lengths[0]])
        assert not torch.equal(out64[lengths[0]:], out_other[lengths[0]:])

        qg = q.detach().requires_grad_(True)
        kg = k.detach().requires_grad_(True)
        vg = v.detach().requires_grad_(True)
        out, lse, _ = run(cu64, qg, kg, vg)
        (out.sum() + lse.sum()).backward()
        assert qg.grad.shape == q.shape
        assert torch.isfinite(qg.grad).all()
        assert torch.isfinite(kg.grad).all()
        assert torch.isfinite(vg.grad).all()

        raw_q = torch.randn(64, 32, 64, device="cuda", dtype=torch.bfloat16)
        raw_k = torch.randn_like(raw_q)
        raw_v = torch.randn_like(raw_q)
        qp, qs = convert_to_mxfp4(raw_q, axis=-1, is_2d_block=True)
        kp, ks = convert_to_mxfp4(raw_k, axis=-1, is_2d_block=True)
        vp, vs = convert_to_mxfp4(raw_v, axis=-1, is_2d_block=True)
        one = torch.tensor([0, 64], device="cuda", dtype=torch.int64)
        kwargs = dict(
            sm_scale=scale, alibi_slopes=None, causal=True, bias=None, dropout_p=0.0, layout="thd",
            max_seqlens_q=64, max_seqlens_k=64, return_scores=False, use_exp2=True, window_left=-1, window_right=0)
        out_i64, lse_i64, _ = attention_mxfp4_forward_triton_impl(
            qp, kp, vp, qs, ks, vs, cu_seqlens_q=one, cu_seqlens_k=one, **kwargs)
        out_i32, lse_i32, _ = attention_mxfp4_forward_triton_impl(
            qp, kp, vp, qs, ks, vs, cu_seqlens_q=one.to(torch.int32), cu_seqlens_k=one.to(torch.int32), **kwargs)
        assert torch.equal(out_i64, out_i32)
        assert torch.equal(lse_i64, lse_i32)
        assert torch.isfinite(out_i64.float()).all()
    finally:
        HadamardFactory.create_transform = raw


def test_varlen_rejects_sequence_not_aligned_to_quant_block():
    torch.manual_seed(0)
    heads, dim = 4, 64
    q = torch.randn(96, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 48, 96], device="cuda", dtype=torch.int32)
    with pytest.raises(AssertionError, match="quant block size"):
        triton_attention_mxfp4(
            q, k, v, None, None, dim**-0.5, 0.0, cu, cu, 48, 48, True, False, True, "thd",
            window_size=(-1, 0))


def test_attention_honors_hadamard_uos_and_sr_grad():
    from alto.kernels.hadamard_transform.transform import HadamardFactory

    class _Scale:
        def __init__(self):
            self.calls = 0

        def __call__(self, value, inverse=False, left_mul=False):
            del inverse, left_mul
            self.calls += 1
            return value * 3

    torch.manual_seed(0)
    heads, dim = 4, 64
    q = torch.randn(64, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    cu = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    scale = dim**-0.5
    transform = _Scale()
    raw = HadamardFactory.__dict__["create_transform"]
    HadamardFactory.create_transform = classmethod(lambda cls, device: transform)
    try:
        def run(**flags):
            query = q.detach().requires_grad_(True)
            out, _, _ = triton_attention_mxfp4(
                query, k, v, None, None, scale, 0.0, cu, cu, 64, 64, True, False, True, "thd",
                window_size=(-1, 0), **flags)
            return out, query

        plain, _ = run(use_hadamard=False, use_uos=False, use_sr_grad=False)
        assert transform.calls == 0
        uos, _ = run(use_hadamard=False, use_uos=True, use_sr_grad=False)
        assert not torch.equal(plain, uos)
        rotated, query = run(use_hadamard=True, use_uos=False, use_sr_grad=True)
        assert transform.calls > 0
        assert not torch.equal(plain, rotated)
        # A constant upstream gradient is an exact FP4 value, so stochastic
        # rounding leaves it unchanged. A random gradient crosses bin edges.
        grad_out = torch.randn_like(rotated)
        rotated.backward(grad_out)
        again, query_again = run(use_hadamard=True, use_uos=False, use_sr_grad=True)
        again.backward(grad_out)
        assert torch.isfinite(query.grad).all()
        assert torch.isfinite(query_again.grad).all()
        assert not torch.equal(query.grad, query_again.grad)
    finally:
        HadamardFactory.create_transform = raw
