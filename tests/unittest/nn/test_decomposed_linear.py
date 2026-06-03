# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from alto.nn import DecomposedLinear
from alto.kernels.dispatch import TrainingOpConfig, swap_params
from alto.kernels.dispatch.tensor import TrainingWeightWrapperBaseTensor


# ---------------------------------------------------------------------------
# Shared scheme spec + helpers
#
# Per-scheme quirks (block size, two-level-scaling default, SNR bar) live in a
# single table so the test bodies stay scheme-agnostic and NVFP4/MXFP4 are
# exercised apples-to-apples. Adding a future scheme = one new row here.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchemeTestSpec:
    block_size: int
    two_level_scaling: str
    min_snr: float


SCHEME_SPECS = {
    "mxfp4": SchemeTestSpec(block_size=32, two_level_scaling="none", min_snr=10.0),
    "nvfp4": SchemeTestSpec(block_size=16, two_level_scaling="none", min_snr=10.0),
}

QUANT_PRECISIONS = list(SCHEME_SPECS)


def make_quant_config(precision: str, **overrides) -> TrainingOpConfig:
    """Single entry point for building a quant config, encapsulating the
    per-scheme ``two_level_scaling`` semantics (nvfp4: tensorwise-capable,
    mxfp4: blockwise-capable; both default to ``none`` here)."""
    spec = SCHEME_SPECS[precision]
    cfg = dict(
        precision=precision,
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=True,
        use_sr_grad=False,
        use_dge=False,
        two_level_scaling=spec.two_level_scaling,
    )
    cfg.update(overrides)
    return TrainingOpConfig(**cfg)


def quant_batch(precision: str) -> int:
    """Activation rows must be divisible by the scheme block size when
    ``use_2dblock_x=False``; pick a comfortable multiple for a meaningful SNR."""
    return 4 * SCHEME_SPECS[precision].block_size


def build_decomposed_linear(
    in_features, out_features, bias, lora_rank, *, device, init_std=0.1, seed=0
):
    torch.manual_seed(seed)
    dl = DecomposedLinear(in_features, out_features, bias, lora_rank).to(device)
    dl.weight.data.normal_(mean=0, std=init_std)
    if bias:
        dl.bias.data.normal_(mean=0, std=init_std)
    dl.u.data.normal_(mean=0, std=init_std)
    dl.v.data.normal_(mean=0, std=init_std)
    dl.sigma.data.normal_(mean=0, std=init_std)
    return dl


def compute_snr(y_ref, y):
    return 20 * torch.log10(torch.norm(y_ref) / torch.norm(y_ref - y))


# ---------------------------------------------------------------------------
# Math correctness of from_linear (scheme-agnostic, CPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_features", [128, 256, 512])
@pytest.mark.parametrize("out_features", [128, 256, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [16, 32, 64])
def test_decomposed_linear(in_features, out_features, bias, lora_rank):
    STD = 0.1
    linear = nn.Linear(in_features, out_features, bias=bias)
    linear.weight.data.normal_(mean=0, std=STD)
    if bias:
        linear.bias.data.normal_(mean=0, std=STD)
    x = torch.randn(1, in_features)

    decomposed_linear = DecomposedLinear.from_linear(linear, lora_rank=lora_rank)
    decomposed_linear.u.data.normal_(mean=0, std=STD)
    decomposed_linear.v.data.normal_(mean=0, std=STD)
    decomposed_linear.sigma.data.normal_(mean=0, std=STD)
    y_decomposed = decomposed_linear(x)

    combined_weight = decomposed_linear.weight + decomposed_linear.u.T @ torch.diag(decomposed_linear.sigma) @ decomposed_linear.v.T
    y_ref = F.linear(x, combined_weight, bias=decomposed_linear.bias)

    assert torch.allclose(y_ref, y_decomposed, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Meta-device build + swap path (TorchTitan).
#
# TorchTitan builds the model on the "meta" device and on_convert wraps params
# BEFORE to_empty/init_weights materialize them. from_linear must therefore
# allocate u/v/sigma on the source device (meta) -- never on CPU then
# `.to("meta")`, which triggers an incompatible set_data and crashes -- and
# swap_params must tolerate meta tensors. CPU-only (no GPU required).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [16, 32])
@pytest.mark.parametrize("precision", QUANT_PRECISIONS)
def test_decomposed_linear_meta_build_and_swap(bias, lora_rank, precision):
    in_features, out_features = 128, 256
    with torch.device("meta"):
        linear = nn.Linear(in_features, out_features, bias=bias)

    dl = DecomposedLinear.from_linear(linear, lora_rank=lora_rank)
    assert dl.u.is_meta and dl.v.is_meta and dl.sigma.is_meta
    assert dl.weight.is_meta
    assert dl.u.shape == (lora_rank, out_features)
    assert dl.v.shape == (in_features, lora_rank)
    assert dl.sigma.shape == (lora_rank,)

    cfg = make_quant_config(precision)
    for p in ("weight", "u", "v"):
        swap_params(dl, config=cfg, target_parameter_name=p)
        assert isinstance(getattr(dl, p).data, TrainingWeightWrapperBaseTensor)

# ---------------------------------------------------------------------------
# End-to-end quantized forward SNR (CUDA), parametrized over all schemes.
# Feature sizes use only the small/large boundary [128, 512] to keep CI cost
# down while still covering the block-alignment edges.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("in_features", [128, 512])
@pytest.mark.parametrize("out_features", [128, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora_rank", [32, 64])
@pytest.mark.parametrize("precision", QUANT_PRECISIONS)
def test_decomposed_linear_quantization(in_features, out_features, bias, lora_rank, precision):
    spec = SCHEME_SPECS[precision]
    decomposed_linear = build_decomposed_linear(
        in_features, out_features, bias, lora_rank, device="cuda"
    )

    x = torch.randn(quant_batch(precision), in_features, device="cuda")
    y_ref = decomposed_linear(x)

    quant_op_config = make_quant_config(precision)
    for p in ("weight", "u", "v"):
        swap_params(decomposed_linear, config=quant_op_config, target_parameter_name=p)
    y = decomposed_linear(x)

    max_diff = torch.max(torch.abs(y_ref - y))
    mean_diff = torch.mean(torch.abs(y_ref - y))
    snr = compute_snr(y_ref, y)
    cossim = torch.nn.functional.cosine_similarity(y_ref.flatten(), y.flatten(), dim=0)
    print(f"precision={precision}, snr={snr}, cossim={cossim}, max_diff={max_diff}, mean_diff={mean_diff}")
    assert snr > spec.min_snr, f"SNR too low for {precision}: {snr}"
