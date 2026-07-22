# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Plot toy-attention training loss curves (MXFP8 vs blockwise FP8 vs bf16).

Standalone helper, not a pytest test. It mirrors ``test_e2e_attention.py`` so the
curve shows the same teacher-student task that the E2E sanity test asserts on:

    python tests/unittest/mxfp8/plot_e2e_attention_curve.py

Outputs ``e2e_attention_loss_curve.png`` next to this file and prints the full
per-step loss series for all paths.
"""

import os

import torch

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

COLORS = {
    "mxfp8 attention": "tab:blue",
    "blockwise fp8 attention": "tab:green",
    "bf16 baseline": "tab:orange",
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


def _save_plot(series, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; skipped saving curve image.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    for name in PATHS:
        ax.plot(series[name], label=name, color=COLORS[name],
                linestyle="--" if name == "bf16 baseline" else "-")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title("toy attention training")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved curve to {out_path}")


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, seqlen, num_heads, head_dim = 1, 128, 4, 64
    model_dim = num_heads * head_dim
    steps, lr = 40, 0.5
    causal = True

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

    print(f"\n===== toy attention: batch={batch}, seqlen={seqlen}, heads={num_heads}, head_dim={head_dim} =====")
    print(f"{'step':>4} " + " ".join(f"{name:>18}" for name in PATHS))
    for i in range(steps):
        print(f"{i:>4} " + " ".join(f"{series[name][i]:>18.8f}" for name in PATHS))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_attention_loss_curve.png")
    _save_plot(series, out_path)


if __name__ == "__main__":
    main()
