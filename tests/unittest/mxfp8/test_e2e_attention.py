# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Toy attention training sanity for low-precision attention kernels.

The grouped-GEMM E2E test trains a tiny MoE because that kernel's native data
shape is expert-routed matmul. Attention has different data: Q/K/V over a
sequence. This test keeps the same E2E idea, but uses a teacher-student toy
attention task so the signal belongs to the attention kernel, not MoE routing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required.")

from alto.kernels.blockwise_fp8.triton_flash_attention_fp8_block import triton_attention_block
from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8


def _init_weights(model_dim, dtype, device, seed, scale=0.05):
    torch.manual_seed(seed)
    return tuple(
        torch.randn((model_dim, model_dim), dtype=dtype, device=device) * scale
        for _ in range(4)
    )


def _make_student_weights(teacher_weights, seed, noise_scale=0.02):
    torch.manual_seed(seed)
    return tuple(w + torch.randn_like(w) * noise_scale for w in teacher_weights)


def _bf16_attention(q, k, v, sm_scale, causal):
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale)


def _mxfp8_attention(q, k, v, sm_scale, causal):
    seqlen_q = q.shape[2]
    seqlen_k = k.shape[2]
    return triton_attention_mxfp8(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=seqlen_q,
        max_seqlens_k=seqlen_k,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]


def _blockwise_fp8_attention(q, k, v, sm_scale, causal):
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    seqlen_q = q.shape[1]
    seqlen_k = k.shape[1]
    return triton_attention_block(
        q,
        k,
        v,
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=seqlen_q,
        max_seqlens_k=seqlen_k,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bshd",
        use_fp8=True,
    )[0].transpose(1, 2)


PATHS = {
    "mxfp8 attention": _mxfp8_attention,
    "blockwise fp8 attention": _blockwise_fp8_attention,
    "bf16 baseline": _bf16_attention,
}


def _toy_attention(x, weights, attention_fn, num_heads, head_dim, causal):
    wq, wk, wv, wo = weights
    batch, seqlen, model_dim = x.shape
    q = (x @ wq).reshape(batch, seqlen, num_heads, head_dim).transpose(1, 2)
    k = (x @ wk).reshape(batch, seqlen, num_heads, head_dim).transpose(1, 2)
    v = (x @ wv).reshape(batch, seqlen, num_heads, head_dim).transpose(1, 2)

    attn_out = attention_fn(q, k, v, head_dim**-0.5, causal)
    attn_out = attn_out.transpose(1, 2).reshape(batch, seqlen, model_dim)
    return attn_out @ wo


def _train(attention_fn, x, w_init, target, num_heads, head_dim, causal, steps, lr):
    weights = tuple(w.clone().detach().requires_grad_(True) for w in w_init)
    losses = []
    for _ in range(steps):
        out = _toy_attention(x, weights, attention_fn, num_heads, head_dim, causal)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        losses.append(loss.item())
        for w in weights:
            w.grad = None
        loss.backward()
        with torch.no_grad():
            for w in weights:
                w -= lr * w.grad
    return losses


def _window_mean(xs, frac=0.2):
    w = max(1, int(len(xs) * frac))
    return sum(xs[:w]) / w, sum(xs[-w:]) / w


@pytest.mark.parametrize("causal", [True])
def test_toy_attention_trains_and_tracks_bf16(causal):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, seqlen, num_heads, head_dim = 1, 128, 4, 64
    model_dim = num_heads * head_dim
    steps, lr = 40, 0.5

    torch.manual_seed(1234)
    x = torch.randn((batch, seqlen, model_dim), dtype=dtype, device=device) * 0.5
    teacher_weights = _init_weights(model_dim, dtype, device, seed=2345)
    w_init = _make_student_weights(teacher_weights, seed=3456)
    with torch.no_grad():
        target = _toy_attention(x, teacher_weights, _bf16_attention, num_heads, head_dim, causal)

    series = {
        name: _train(fn, x, w_init, target, num_heads, head_dim, causal, steps, lr)
        for name, fn in PATHS.items()
    }

    _, bf16_tail = _window_mean(series["bf16 baseline"])
    for name, losses in series.items():
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            f"{name} loss went non-finite: {losses}"
        head, tail = _window_mean(losses)
        assert tail < head * 0.98, f"{name} loss did not trend down: head={head:.6f} tail={tail:.6f}"
        assert tail < bf16_tail * 2.0, \
            f"{name} final loss diverged from bf16: {name}_tail={tail:.6f} bf16_tail={bf16_tail:.6f}"
