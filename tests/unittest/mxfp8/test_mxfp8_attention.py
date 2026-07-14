# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Forward numerical tests for the MXFP8 (e4m3) flash attention kernel.

Mirrors ``tests/unittest/mxfp4/test_mxfp_attention.py``. The MXFP8 kernel is
forward-only (backward is not implemented), so only the forward output is
validated against a bf16 SDPA reference via SNR / cosine-similarity.
"""

import pytest
from tabulate import tabulate
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required.")

from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8
from .utils import calc_snr, calc_cossim


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout="bshd"):
    """Compute reference output using PyTorch's built-in SDPA."""
    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    o_ref = torch.nn.functional.scaled_dot_product_attention(q,
                                                             k,
                                                             v,
                                                             is_causal=causal,
                                                             scale=sm_scale,
                                                             enable_gqa=n_rep > 1)
    if layout == "bshd":
        o_ref = o_ref.transpose(1, 2)
    return o_ref


class AttnConfig:

    def __init__(self, seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v):
        self.seqlen_q = seqlen_q
        self.seqlen_kv = seqlen_kv
        self.num_head_q = num_head_q
        self.num_head_kv = num_head_kv
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v


test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=2048, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True])
def test_attention(batch, config, causal):
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, seqlen_kv, num_head_q, num_head_kv, head_dim_qk, head_dim_v = (
        config.seqlen_q,
        config.seqlen_kv,
        config.num_head_q,
        config.num_head_kv,
        config.head_dim_qk,
        config.head_dim_v,
    )
    q_layout = (batch, seqlen_q, num_head_q, head_dim_qk)
    k_layout = (batch, seqlen_kv, num_head_kv, head_dim_qk)
    v_layout = (batch, seqlen_kv, num_head_kv, head_dim_v)

    torch.manual_seed(1234)

    query = torch.randn(q_layout, device=device, dtype=dtype, requires_grad=True)
    key = torch.randn(k_layout, device=device, dtype=dtype, requires_grad=True)
    value = torch.randn(v_layout, device=device, dtype=dtype, requires_grad=True)
    query_ref = query.clone().detach().requires_grad_()
    key_ref = key.clone().detach().requires_grad_()
    value_ref = value.clone().detach().requires_grad_()

    sm_scale = query.shape[-1]**(-0.5)
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)

    o = triton_attention_mxfp8(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=seqlen_q,
        max_seqlens_k=seqlen_kv,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0].transpose(1, 2)

    output_snr = calc_snr(o_ref, o)
    output_sim = calc_cossim(o_ref, o)
    print()
    print(tabulate([
        ["O", output_snr, output_sim],
    ], headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    assert output_sim > 0.99, f"output cosine-sim too low: {output_sim}"
    assert output_snr > 20, f"output SNR too low: {output_snr}"
