# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import torch
import yaml
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

from alto.config import TrainingOpConfig
from alto.models.gpt_oss.config_registry import (
    MXFP8_ATTN_RECIPE,
    gpt_oss_debugmodel_flex,
)
from alto.modifiers.lpt.base import LowPrecisionTrainingModifier
from alto.nn.lpt import LPTModule, lpt_module
from torchtitan.models.common.attention import (
    FlexInnerAttention,
    SlidingWindowFlexInnerAttention,
)
from torchtitan.models.gpt_oss.model import apply_attention_sink_rescale
from torchtitan.models.kimi_k2_7.qk_clip import QKClipFlexInnerAttention

from .utils import calc_cossim, calc_snr

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required.")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _mxfp8_config(consistent_delta: bool = False) -> TrainingOpConfig:
    return TrainingOpConfig(
        precision="mxfp8_e4m3",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
        attention_consistent_delta=consistent_delta,
    )


def _recipe_modifier() -> LowPrecisionTrainingModifier:
    recipe = yaml.safe_load((_REPO_ROOT / MXFP8_ATTN_RECIPE).read_text())
    args = recipe["training_stage"]["lpt_modifiers"]["LowPrecisionTrainingModifier"]
    return LowPrecisionTrainingModifier(**args)


def _packed_positions(doc_lengths: list[int], device: str) -> torch.Tensor:
    return torch.cat([torch.arange(n, device=device) for n in doc_lengths])


def test_lpt_flex_has_no_weights_and_requires_mxfp8():
    module_cls = lpt_module(FlexInnerAttention)
    config = _mxfp8_config()
    module = module_cls(module_cls.Config(training_op_config=config))
    assert isinstance(module, LPTModule)
    assert list(module.iter_lpt_weights()) == []
    assert "training_op_config=TrainingOpConfig(precision='mxfp8_e4m3'" in repr(module)

    mxfp4 = TrainingOpConfig(
        precision="mxfp4",
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
    )
    with pytest.raises(ValueError, match="Unsupported Flex precision"):
        module_cls(module_cls.Config(training_op_config=mxfp4))


def test_lpt_flex_rejects_subclasses_that_need_extra_aux():
    with pytest.raises(TypeError, match="extra Flex aux outputs"):
        lpt_module(QKClipFlexInnerAttention)


def test_gpt_oss_mxfp8_attn_recipe_installs_lpt_flex_modules():
    modifier = _recipe_modifier()
    model_config = gpt_oss_debugmodel_flex().model
    modifier.on_convert_config(model_config)

    sliding, full = model_config.layers[0].attention, model_config.layers[1].attention
    assert type(sliding.inner_attention).__qualname__ == (
        "LPTSlidingWindowFlexInnerAttention.Config"
    )
    assert sliding.inner_attention.sliding_window_size == 128
    assert type(full.inner_attention).__qualname__ == "LPTFlexInnerAttention.Config"
    assert full.inner_attention.training_op_config.attention_consistent_delta is True
    assert type(full.qkv_linear.wqkv).__qualname__ == "ColumnParallelLinear.Config"

    with torch.device("meta"):
        model = model_config.build()
    assert modifier.on_convert(model)


def _run_attention(module, q, k, v, sinks, metadata, scale):
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, sinks)]
    q_, k_, v_, sinks_ = leaves
    out = module(
        q_, k_, v_,
        attention_metadata=metadata,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
        out_transform=lambda o, lse: apply_attention_sink_rescale(o, lse, sinks_),
    )
    return out, leaves


def _bf16_flex_reference(q, k, v, sinks, metadata, scale):
    """FlexInnerAttention's math, run eagerly instead of through max-autotune compile."""
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, k, v, sinks)]
    q_, k_, v_, sinks_ = leaves
    out, aux = flex_attention(
        *(t.transpose(0, 1).unsqueeze(0) for t in (q_, k_, v_)),
        block_mask=metadata,
        scale=scale,
        enable_gqa=q.shape[1] != k.shape[1],
        return_aux=AuxRequest(lse=True),
    )
    out_THV = out.squeeze(0).transpose(0, 1)
    lse_TH = aux.lse.squeeze(0).transpose(0, 1)
    return apply_attention_sink_rescale(out_THV, lse_TH, sinks_), leaves


@cuda_only
@pytest.mark.parametrize("sliding_window_size", [None, 128])
def test_lpt_flex_matches_bf16_flex_with_sinks(sliding_window_size):
    """GPT-OSS's exact call: packed docs, optional window, sinks via out_transform."""
    torch.manual_seed(2468)
    num_head_q, num_head_kv, head_dim = 8, 2, 64
    positions = _packed_positions([300, 212, 512], device="cuda")
    seqlen = positions.shape[0]
    if sliding_window_size is None:
        base, extra = FlexInnerAttention, {}
    else:
        base, extra = SlidingWindowFlexInnerAttention, {"sliding_window_size": sliding_window_size}
    module_cls = lpt_module(base)
    module_config = module_cls.Config(training_op_config=_mxfp8_config(), **extra)
    module = module_cls(module_config)
    metadata = module_config.build_attention_metadata(positions)

    q = torch.randn(seqlen, num_head_q, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seqlen, num_head_kv, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn(num_head_q, device="cuda", dtype=torch.float32)
    do = torch.randn_like(q)
    scale = head_dim**-0.5

    out, leaves = _run_attention(module, q, k, v, sinks, metadata, scale)
    (out * do).sum().backward()
    ref_out, ref_leaves = _bf16_flex_reference(q, k, v, sinks, metadata, scale)
    (ref_out * do).sum().backward()

    assert out.shape == ref_out.shape
    pairs = [("o", ref_out, out)] + [
        (name, ref.grad, got.grad)
        for name, ref, got in zip(("dQ", "dK", "dV", "dSinks"), ref_leaves, leaves)
    ]
    for name, expected, actual in pairs:
        assert calc_cossim(expected, actual) > 0.99, name
        assert calc_snr(expected, actual) > 18, name


@cuda_only
def test_lpt_flex_forwards_consistent_delta():
    """dV = Pᵀ @ dO never reads delta; only dQ/dK move when the flag reaches the kernel."""
    torch.manual_seed(3456)
    positions = _packed_positions([256], device="cuda")
    q = torch.randn(256, 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    do = torch.randn_like(q)

    def grads(consistent_delta):
        module_cls = lpt_module(FlexInnerAttention)
        module_config = module_cls.Config(training_op_config=_mxfp8_config(consistent_delta))
        metadata = module_config.build_attention_metadata(positions)
        leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
        out = module_cls(module_config)(*leaves, attention_metadata=metadata)
        (out * do).sum().backward()
        return [t.grad for t in leaves]

    (dq_off, dk_off, dv_off), (dq_on, dk_on, dv_on) = grads(False), grads(True)
    torch.testing.assert_close(dv_off, dv_on, rtol=0, atol=0)
    assert not torch.equal(dq_off, dq_on)
    assert not torch.equal(dk_off, dk_on)
