# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import inspect
from dataclasses import replace

import pytest
import torch
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
    SlidingWindowVarlenInnerAttention,
    VarlenInnerAttention,
)
from torchtitan.models.gpt_oss.model import apply_attention_sink_rescale

from alto.kernels.fp4.testing_utils import calc_cossim
from alto.models.build_record import rebuild_recorded_model
from alto.models.gpt_oss.config_registry import (
    gpt_oss_20b_pretrain,
    gpt_oss_20b_pretrain_flex_detached_lse,
)
from alto.nn.attention import (
    DetachedLseFlexInnerAttention,
    DetachedLseSlidingWindowFlexInnerAttention,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required.")


def test_gpt_oss_20b_flex_detached_lse_only_changes_attention():
    config = gpt_oss_20b_pretrain_flex_detached_lse()
    built = config.model
    rebuild_recorded_model(config)
    assert config.model is not built

    for layer_id, layer in enumerate(config.model.layers):
        inner = layer.attention.inner_attention
        if layer_id % 2 == 0:
            assert type(inner) is DetachedLseSlidingWindowFlexInnerAttention.Config, layer_id
            assert inner.sliding_window_size == 128
        else:
            assert type(inner) is DetachedLseFlexInnerAttention.Config, layer_id

    reference = gpt_oss_20b_pretrain()
    for name in ("training", "optim", "parallelism", "loss", "checkpointer", "debug", "metrics"):
        assert getattr(config, name) == getattr(reference, name), name
    for ours, ref in (
        (config.dataloader, reference.dataloader),
        (config.validator.dataloader, reference.validator.dataloader),
    ):
        assert _strip_filter_lambdas(ours) == _strip_filter_lambdas(ref)
    assert (config.validator.freq, config.validator.steps) == (
        reference.validator.freq,
        reference.validator.steps,
    )
    assert config.dataloader.dataset.dataset.processor.seq_align is None
    assert config.training.steps == 1_200_000


def _strip_filter_lambdas(loader):
    """``megatron_dataset_config`` builds a new filter lambda per call; drop it before comparing."""
    packing = loader.dataset
    return replace(loader, dataset=replace(packing, dataset=replace(packing.dataset, post_filters=())))


def _positional_names(cls) -> list[str]:
    parameters = inspect.signature(cls.forward).parameters.values()
    return [p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD]


def test_detached_lse_forward_keeps_flex_positional_names():
    """TorchTitan reshards q/k/v by these names; without them the inputs are dropped."""
    for cls in (DetachedLseFlexInnerAttention, DetachedLseSlidingWindowFlexInnerAttention):
        assert _positional_names(cls) == _positional_names(FlexInnerAttention), cls


def _packed_positions(doc_lengths: list[int], device: str) -> torch.Tensor:
    return torch.cat([torch.arange(n, device=device) for n in doc_lengths])


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def _run(config, positions, q, k, v, sinks, do) -> dict[str, torch.Tensor]:
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v, sinks)]
    q_, k_, v_, sinks_ = leaves
    out = config.build()(
        q_, k_, v_,
        attention_metadata=config.build_attention_metadata(positions),
        scale=q.shape[-1] ** -0.5,
        enable_gqa=True,
        out_transform=lambda o, lse: apply_attention_sink_rescale(o, lse, sinks_),
    )
    (out * do).sum().backward()
    return dict(zip(("o", "dQ", "dK", "dV", "dSinks"), [out, *(t.grad for t in leaves)]))


@cuda_only
@pytest.mark.parametrize("sliding_window_size", [None, 128])
def test_detached_lse_flex_trains_like_varlen(sliding_window_size):
    """GPT-OSS's call: packed documents, optional window, sinks reading the LSE."""
    if sliding_window_size is None:
        configs = (
            VarlenInnerAttention.Config(),
            FlexInnerAttention.Config(),
            DetachedLseFlexInnerAttention.Config(),
        )
    else:
        window = {"sliding_window_size": sliding_window_size}
        configs = (
            SlidingWindowVarlenInnerAttention.Config(**window),
            SlidingWindowFlexInnerAttention.Config(**window),
            DetachedLseSlidingWindowFlexInnerAttention.Config(**window),
        )
    torch.manual_seed(1357)
    positions = _packed_positions([300, 212, 512], device="cuda")
    seqlen = positions.shape[0]
    # Peaked attention and sinks near the LSE make the LSE term of dQ/dK
    # large next to BF16 noise.
    q = 2 * torch.randn(seqlen, 8, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen, 2, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = 6 + torch.randn(8, device="cuda", dtype=torch.float32)
    do = torch.randn_like(q)

    varlen, flex, detached = (_run(c, positions, q, k, v, sinks, do) for c in configs)

    for name in ("o", "dV", "dSinks"):
        torch.testing.assert_close(detached[name], flex[name], rtol=0, atol=0)
    for name in ("o", "dQ", "dK", "dV", "dSinks"):
        assert calc_cossim(varlen[name], detached[name]) > 0.999, name
    for name in ("dQ", "dK"):
        assert _rel_err(detached[name], varlen[name]) < 0.25 * _rel_err(flex[name], varlen[name]), name
