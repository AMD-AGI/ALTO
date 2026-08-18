# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Shared toy-attention E2E helpers for MXFP8 training sanity checks."""

from dataclasses import asdict, dataclass

import torch

from alto.kernels.blockwise_fp8.triton_flash_attention_fp8_block import triton_attention_block
from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8


@dataclass(frozen=True)
class AttentionExperimentConfig:
    batch: int = 1
    seqlen: int = 128
    num_heads: int = 4
    head_dim: int = 64
    steps: int = 40
    lr: float = 0.5
    causal: bool = True
    input_scale: float = 0.5
    weight_scale: float = 0.05
    student_noise_scale: float = 0.02

    @property
    def model_dim(self):
        return self.num_heads * self.head_dim

    def to_json(self):
        data = asdict(self)
        data["model_dim"] = self.model_dim
        data["dtype"] = "torch.bfloat16"
        return data


DEFAULT_SEEDS = (1234, 2234, 3234, 4234, 5234)


def _init_weights(model_dim, dtype, device, seed, scale):
    torch.manual_seed(seed)
    return tuple(
        torch.randn((model_dim, model_dim), dtype=dtype, device=device) * scale
        for _ in range(4)
    )


def _make_student_weights(teacher_weights, seed, noise_scale):
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


def _toy_attention(x, weights, attention_fn, config):
    wq, wk, wv, wo = weights
    batch, seqlen, model_dim = x.shape
    q = (x @ wq).reshape(batch, seqlen, config.num_heads, config.head_dim).transpose(1, 2)
    k = (x @ wk).reshape(batch, seqlen, config.num_heads, config.head_dim).transpose(1, 2)
    v = (x @ wv).reshape(batch, seqlen, config.num_heads, config.head_dim).transpose(1, 2)

    attn_out = attention_fn(q, k, v, config.head_dim**-0.5, config.causal)
    attn_out = attn_out.transpose(1, 2).reshape(batch, seqlen, model_dim)
    return attn_out @ wo


def make_case(config, seed, device, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    x = torch.randn((config.batch, config.seqlen, config.model_dim),
                    dtype=dtype, device=device) * config.input_scale
    teacher_weights = _init_weights(
        config.model_dim, dtype, device, seed + 10000, config.weight_scale)
    w_init = _make_student_weights(
        teacher_weights, seed + 20000, config.student_noise_scale)
    with torch.no_grad():
        target = _toy_attention(x, teacher_weights, _bf16_attention, config)
    return x, w_init, target


def train_path(attention_fn, x, w_init, target, config):
    weights = tuple(w.clone().detach().requires_grad_(True) for w in w_init)
    losses = []
    for _ in range(config.steps):
        out = _toy_attention(x, weights, attention_fn, config)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        losses.append(float(loss.item()))
        for w in weights:
            w.grad = None
        loss.backward()
        with torch.no_grad():
            for w in weights:
                w -= config.lr * w.grad
    return losses


def run_training_experiment(config, seeds=DEFAULT_SEEDS, device=None):
    device = device or torch.device("cuda")
    per_path = {name: {} for name in PATHS}
    for seed in seeds:
        x, w_init, target = make_case(config, seed, device)
        for name, fn in PATHS.items():
            per_path[name][str(seed)] = train_path(fn, x, w_init, target, config)
    return summarize_experiment(config, seeds, per_path)


def run_forward_only_experiment(config, seeds=DEFAULT_SEEDS, device=None):
    device = device or torch.device("cuda")
    per_path = {name: {} for name in PATHS}
    for seed in seeds:
        x, w_init, target = make_case(config, seed, device)
        weights = tuple(w.clone().detach().requires_grad_(True) for w in w_init)
        seed_series = {name: [] for name in PATHS}
        for _ in range(config.steps):
            with torch.no_grad():
                frozen_weights = tuple(w.detach() for w in weights)
                for name, fn in PATHS.items():
                    out = _toy_attention(x, frozen_weights, fn, config)
                    loss = torch.nn.functional.mse_loss(out.float(), target.float())
                    seed_series[name].append(float(loss.item()))

            out = _toy_attention(x, weights, _bf16_attention, config)
            loss = torch.nn.functional.mse_loss(out.float(), target.float())
            for w in weights:
                w.grad = None
            loss.backward()
            with torch.no_grad():
                for w in weights:
                    w -= config.lr * w.grad

        for name, losses in seed_series.items():
            per_path[name][str(seed)] = losses
    return summarize_experiment(config, seeds, per_path)


def _mean_std(per_seed):
    series = torch.tensor(list(per_seed.values()), dtype=torch.float32)
    mean = series.mean(dim=0)
    std = series.std(dim=0, unbiased=False)
    return [float(x) for x in mean], [float(x) for x in std]


def summarize_experiment(config, seeds, per_path):
    paths = {}
    for name, per_seed in per_path.items():
        mean, std = _mean_std(per_seed)
        paths[name] = {
            "per_seed": per_seed,
            "mean": mean,
            "std": std,
            "final_mean": mean[-1],
            "final_std": std[-1],
        }
    return {
        "config": config.to_json(),
        "seeds": list(seeds),
        "paths": paths,
    }


def window_mean(xs, frac=0.2):
    w = max(1, int(len(xs) * frac))
    return sum(xs[:w]) / w, sum(xs[-w:]) / w
