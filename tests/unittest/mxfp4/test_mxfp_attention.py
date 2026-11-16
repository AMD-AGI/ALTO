# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import pytest
from tabulate import tabulate
import torch

from alto.kernels.fp4.mxfp4.triton_flash_attention_mxfp4 import triton_attention_mxfp4
from .utils import calc_cossim, calc_snr

_RESULTS = []


def attention_vanilla_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout="bshd"):
    """Compute the bf16 SDPA reference in the same layout as the inputs."""

    if layout == "bshd":
        num_heads = q.shape[2]
        n_kv_heads = k.shape[2]
        n_rep = num_heads // n_kv_heads

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Unknown layout {layout}")

    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=causal,
        scale=sm_scale,
        enable_gqa=n_rep > 1,
    )
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

    def __repr__(self):
        return (
            f"sq{self.seqlen_q}_sk{self.seqlen_kv}_hq{self.num_head_q}_"
            f"hkv{self.num_head_kv}_dqk{self.head_dim_qk}_dv{self.head_dim_v}"
        )


test_cases = [
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=32, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=16, num_head_kv=16, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=128, num_head_kv=128, head_dim_qk=192, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=32, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=1024, seqlen_kv=1024, num_head_q=48, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
    # tl.dot_scaled does not support K < 64, so the packed ASM path rejects head dim 32.
    # AttnConfig(seqlen_q=4096 + 64, seqlen_kv=4096 + 64, num_head_q=2, num_head_kv=1, head_dim_qk=32, head_dim_v=32),
    AttnConfig(seqlen_q=2048, seqlen_kv=2048, num_head_q=64, num_head_kv=8, head_dim_qk=128, head_dim_v=128),
]

_HEADERS = ["Tensor", "SNR", "Cosine Sim"]
_SUMMARY_HEADERS = [
    "Sq", "Sk", "Hq", "Hkv", "Dqk", "Dv", "causal", "sparse",
    "O SNR", "O cos", "dQ SNR", "dQ cos", "dK SNR", "dK cos", "dV SNR", "dV cos",
]
# Floors sit just under the MXFP4-vs-bf16-SDPA numbers measured on this suite.
_DENSE_FLOORS = {"O": (10.5, 0.96), "dQ": (10.5, 0.96), "dK": (12.0, 0.97), "dV": (25.0, 0.999)}
_SPARSE_FLOORS = {"O": (10.5, 0.96), "dQ": (10.0, 0.95), "dK": (10.0, 0.95), "dV": (12.0, 0.97)}


def _metrics(reference, actual):
    return calc_snr(reference, actual), calc_cossim(reference, actual)


def _print_case(title, rows):
    print()
    print(title)
    print(tabulate(rows, headers=_HEADERS, tablefmt="github", floatfmt=".4f"))


def _assert_floors(recorded, floors):
    for name, (min_snr, min_cosine) in floors.items():
        snr, cosine = recorded[name]
        assert snr > min_snr and cosine > min_cosine, f"{name} snr {snr:.4f} cosine {cosine:.4f}"


def _run_case(batch, config, causal, grad_output=None):
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
    query_ref = query.detach().clone().requires_grad_()
    key_ref = key.detach().clone().requires_grad_()
    value_ref = value.detach().clone().requires_grad_()

    sm_scale = head_dim_qk**-0.5
    o_ref = attention_vanilla_forward_pytorch_ref_impl(query_ref, key_ref, value_ref, sm_scale, causal)
    if grad_output is None:
        o_ref.sum().backward()
    else:
        o_ref.backward(grad_output)

    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    output = triton_attention_mxfp4(
        q,
        k,
        v,
        None,
        None,
        sm_scale,
        0.0,
        0,
        0,
        seqlen_q,
        seqlen_kv,
        causal,
        False,
        True,
        "bhsd",
    )[0].transpose(1, 2)
    if grad_output is None:
        output.sum().backward()
    else:
        output.backward(grad_output)

    pairs = (
        ("O", o_ref, output),
        ("dQ", query_ref.grad, query.grad),
        ("dK", key_ref.grad, key.grad),
        ("dV", value_ref.grad, value.grad),
    )
    rows = []
    recorded = {}
    for name, reference, actual in pairs:
        assert actual is not None, f"{name} grad is missing"
        assert torch.isfinite(actual).all(), f"{name} is not finite"
        snr, cosine = _metrics(reference, actual)
        rows.append([name, snr, cosine])
        recorded[name] = (snr, cosine)
    title = (
        f"batch={batch} seq={seqlen_q}x{seqlen_kv} heads={num_head_q}/{num_head_kv} "
        f"dim={head_dim_qk}/{head_dim_v} causal={causal} sparse_do={grad_output is not None}"
    )
    _print_case(title, rows)
    _RESULTS.append([
        seqlen_q,
        seqlen_kv,
        num_head_q,
        num_head_kv,
        head_dim_qk,
        head_dim_v,
        causal,
        grad_output is not None,
        recorded["O"][0],
        recorded["O"][1],
        recorded["dQ"][0],
        recorded["dQ"][1],
        recorded["dK"][0],
        recorded["dK"][1],
        recorded["dV"][0],
        recorded["dV"][1],
    ])
    return recorded


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention(batch, config, causal):
    _assert_floors(_run_case(batch, config, causal), _DENSE_FLOORS)


@pytest.mark.parametrize("batch", [4])
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_attention_with_sparse_do(batch, config, causal):
    device = "cuda"
    dtype = torch.bfloat16
    do_shape = (batch, config.seqlen_q, config.num_head_q, config.head_dim_v)
    torch.manual_seed(1234)
    grad_output = torch.randn(do_shape, device=device, dtype=dtype) * 1e-3
    mask_token = (torch.randn(do_shape[:-2], device=device) > 0.9).unsqueeze(-1).unsqueeze(-1)
    mask_head = (torch.randn(do_shape[:-1], device=device) > 0.9).unsqueeze(-1)
    grad_output = grad_output * mask_token * mask_head
    _assert_floors(_run_case(batch, config, causal, grad_output=grad_output.to(dtype)), _SPARSE_FLOORS)
