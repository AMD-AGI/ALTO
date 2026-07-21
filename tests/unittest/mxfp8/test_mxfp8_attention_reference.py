# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Golden-reference validation for the MXFP8 (e4m3) flash attention kernel.

Additive to ``test_mxfp8_attention.py`` (which mirrors the mxfp4 attention test:
kernel vs bf16 SDPA). This file adds the two layers that the pure-PyTorch golden
reference enables — a strengthening over the mxfp4 reference, which has neither a
golden reference nor asserts:

  * Layer 1 ``test_reference_matches_bf16_sdpa`` — golden reference vs bf16 SDPA.
    Pure PyTorch, **runs on CPU / any device without CDNA4**; validates the
    algorithm and mxfp8 quantization placement *before* the hardware arrives.
  * Layer 2 ``test_kernel_matches_reference`` — kernel vs golden reference.
    Isolates Triton port bugs (masking / online-softmax / LSE) from the mxfp8
    quantization error itself. Requires CDNA4 (native ``tl.dot_scaled``).

(Layer 3 — kernel vs bf16 SDPA, total end-to-end error — is the committed
``test_mxfp8_attention.py::test_attention``; not duplicated here.)
"""

import pytest
from tabulate import tabulate
import torch

from .utils import (
    calc_snr,
    calc_cossim,
    mxfp8_attention_forward_reference,
    mxfp8_attention_backward_reference,
    mxfp8_attention_backward_reference_stage2,
)
# Reuse the committed shape grid so Layer 2 stays in lock-step with Layer 3.
from .test_mxfp8_attention import AttnConfig, test_cases

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm device is required.")


def _sdpa_bf16_bhsd(q, k, v, sm_scale, causal):
    """bf16 SDPA reference in bhsd layout (the native SDPA head layout)."""
    n_rep = q.shape[1] // k.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1)


def _make_qkv_bhsd(batch, config, device, dtype):
    torch.manual_seed(1234)
    q = torch.randn((batch, config.num_head_q, config.seqlen_q, config.head_dim_qk), device=device, dtype=dtype)
    k = torch.randn((batch, config.num_head_kv, config.seqlen_kv, config.head_dim_qk), device=device, dtype=dtype)
    v = torch.randn((batch, config.num_head_kv, config.seqlen_kv, config.head_dim_v), device=device, dtype=dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# Layer 1 — golden reference vs bf16 SDPA (pure PyTorch, runs anywhere, no CDNA4)
# ---------------------------------------------------------------------------

# Small shapes: the reference loops over key blocks in Python, keep it cheap for CPU.
reference_cases = [
    AttnConfig(seqlen_q=128, seqlen_kv=128, num_head_q=4, num_head_kv=4, head_dim_qk=64, head_dim_v=64),
    AttnConfig(seqlen_q=256, seqlen_kv=256, num_head_q=8, num_head_kv=2, head_dim_qk=128, head_dim_v=128),  # GQA
    AttnConfig(seqlen_q=128, seqlen_kv=256, num_head_q=4, num_head_kv=4, head_dim_qk=128, head_dim_v=128),  # seqlen_kv > seqlen_q
]


@pytest.mark.parametrize("config", reference_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_reference_matches_bf16_sdpa(config, causal):
    """The pure-PyTorch golden reference must track bf16 SDPA within mxfp8 error.

    Runnable on CPU without CDNA4 — the check we can do *before* the hardware
    arrives, to de-risk the algorithm and quantization placement.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    q, k, v = _make_qkv_bhsd(2, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    o_ref = _sdpa_bf16_bhsd(q, k, v, sm_scale, causal)
    o_golden, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal)

    snr = calc_snr(o_ref, o_golden)
    sim = calc_cossim(o_ref, o_golden)
    print()
    print(tabulate([["O (golden vs bf16)", snr, sim]], headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    assert sim > 0.99, f"golden reference cosine-sim vs bf16 SDPA too low: {sim}"
    assert snr > 15, f"golden reference SNR vs bf16 SDPA too low: {snr}"


# ---------------------------------------------------------------------------
# Layer 2 — kernel vs golden reference (requires CDNA4 native tl.dot_scaled)
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True])
def test_kernel_matches_reference(config, causal):
    """Kernel vs golden reference — isolates Triton port bugs from quant error.

    Both sides apply the same mxfp8 quantization, so a large gap here points at a
    Triton port bug (masking / online-softmax / LSE / strides) rather than
    quantization error.
    """
    from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8

    device = "cuda"
    dtype = torch.bfloat16
    q, k, v = _make_qkv_bhsd(4, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    o_kernel = triton_attention_mxfp8(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=config.seqlen_q,
        max_seqlens_k=config.seqlen_kv,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]
    o_golden, _ = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal)

    snr = calc_snr(o_golden, o_kernel)
    sim = calc_cossim(o_golden, o_kernel)
    print()
    print(tabulate([["O (kernel vs golden)", snr, sim]], headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    # Threshold placeholder — calibrate on the first CDNA4 run.
    assert sim > 0.99, f"kernel vs golden cosine-sim too low: {sim}"
    assert snr > 30, f"kernel vs golden SNR too low: {snr}"


# ---------------------------------------------------------------------------
# Backward Layer 1 — golden reference vs autograd fp32 SDPA (pure PyTorch, CPU)
# ---------------------------------------------------------------------------

def _make_do_bhsd(batch, config, device, dtype):
    torch.manual_seed(4321)
    return torch.randn((batch, config.num_head_q, config.seqlen_q, config.head_dim_v), device=device, dtype=dtype)


@pytest.mark.parametrize("config", reference_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_backward_reference_matches_sdpa(config, causal):
    """Backward golden reference (mxfp8 quant) must track autograd fp32 SDPA grads.

    Both sides use top-left causal, so this is valid for non-square shapes too.
    The gap is the mxfp8 quantization error in the gradients; runs on CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    n_rep = config.num_head_q // config.num_head_kv

    q, k, v = _make_qkv_bhsd(2, config, device, dtype)
    do = _make_do_bhsd(2, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    # Autograd fp32 SDPA backward — the "ideal" (unquantized) gradient.
    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    o_auto = torch.nn.functional.scaled_dot_product_attention(
        qg, kg, vg, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1)
    o_auto.backward(do)

    # Golden reference backward — quantized operands, forward-ref o / lse.
    o_ref, lse_ref = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal)
    dq, dk, dv = mxfp8_attention_backward_reference(q, k, v, do, o_ref, lse_ref, sm_scale, causal)

    rows = []
    for name, ref, got in [("dQ", qg.grad, dq), ("dK", kg.grad, dk), ("dV", vg.grad, dv)]:
        rows.append([f"{name} (golden vs sdpa)", calc_snr(ref, got), calc_cossim(ref, got)])
    print()
    print(tabulate(rows, headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    for name, ref, got in [("dQ", qg.grad, dq), ("dK", kg.grad, dk), ("dV", vg.grad, dv)]:
        sim = calc_cossim(ref, got)
        snr = calc_snr(ref, got)
        assert sim > 0.97, f"{name} golden-vs-sdpa cosine-sim too low: {sim}"
        assert snr > 8, f"{name} golden-vs-sdpa SNR too low: {snr}"


# ---------------------------------------------------------------------------
# Backward Stage-2 (A1) Layer 1 — full-quant reference vs autograd fp32 SDPA
#   Adds dO/P/dS quantization on top of Stage-1 (a numerical model of the
#   kernel's tl.dot_scaled). q/k/v reuse the forward 2D-block dequant (A1, no
#   double quant). The gap vs SDPA is the *full* mxfp8 backward quant error —
#   the signal that all-e4m3 backward is viable. Pure PyTorch, runs on CPU/MI250.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("config", reference_cases)
@pytest.mark.parametrize("causal", [True, False])
def test_backward_stage2_reference_matches_sdpa(config, causal):
    """Stage-2 (A1, all-operand-quantized) backward reference vs autograd fp32 SDPA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    n_rep = config.num_head_q // config.num_head_kv

    q, k, v = _make_qkv_bhsd(2, config, device, dtype)
    do = _make_do_bhsd(2, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    vg = v.clone().requires_grad_(True)
    o_auto = torch.nn.functional.scaled_dot_product_attention(
        qg, kg, vg, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1)
    o_auto.backward(do)

    o_ref, lse_ref = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal)
    dq, dk, dv = mxfp8_attention_backward_reference_stage2(q, k, v, do, o_ref, lse_ref, sm_scale, causal)

    rows = []
    for name, ref, got in [("dQ", qg.grad, dq), ("dK", kg.grad, dk), ("dV", vg.grad, dv)]:
        rows.append([f"{name} (stage2 vs sdpa)", calc_snr(ref, got), calc_cossim(ref, got)])
    print()
    print(tabulate(rows, headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    # A1 has no double-quant, so it should land at/above the retired option-A
    # baseline (cossim~0.998 / SNR~23-26 dB). Thresholds keep margin.
    for name, ref, got in [("dQ", qg.grad, dq), ("dK", kg.grad, dk), ("dV", vg.grad, dv)]:
        sim = calc_cossim(ref, got)
        snr = calc_snr(ref, got)
        assert sim > 0.995, f"{name} stage2-vs-sdpa cosine-sim too low: {sim}"
        assert snr > 18, f"{name} stage2-vs-sdpa SNR too low: {snr}"


# ---------------------------------------------------------------------------
# Public autograd path — user-facing forward + backward wrapper
# ---------------------------------------------------------------------------

public_autograd_cases = [
    AttnConfig(seqlen_q=128, seqlen_kv=128, num_head_q=4, num_head_kv=4, head_dim_qk=128, head_dim_v=128),
    AttnConfig(seqlen_q=128, seqlen_kv=128, num_head_q=8, num_head_kv=2, head_dim_qk=128, head_dim_v=128),  # GQA
]


@cuda_only
@pytest.mark.parametrize("config", public_autograd_cases)
@pytest.mark.parametrize("causal", [True])
def test_public_autograd_matches_sdpa(config, causal):
    """Public ``triton_attention_mxfp8`` forward/backward must wire gradients correctly.

    The lower-level backward op is tested across the large shape grid below. This
    smaller test covers the user-facing autograd wrapper: saved tensors, ctx
    metadata, backward op dispatch, and returned gradient positions.
    """
    from alto.kernels.mxfp8.triton_flash_attention_mxfp8 import triton_attention_mxfp8

    device = "cuda"
    dtype = torch.bfloat16
    n_rep = config.num_head_q // config.num_head_kv

    q, k, v = _make_qkv_bhsd(1, config, device, dtype)
    do = _make_do_bhsd(1, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    q_ref = q.clone().detach().requires_grad_(True)
    k_ref = k.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    o_ref = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=causal, scale=sm_scale, enable_gqa=n_rep > 1)
    o_ref.backward(do)

    q_kernel = q.clone().detach().requires_grad_(True)
    k_kernel = k.clone().detach().requires_grad_(True)
    v_kernel = v.clone().detach().requires_grad_(True)
    o_kernel = triton_attention_mxfp8(
        q_kernel.contiguous(),
        k_kernel.contiguous(),
        v_kernel.contiguous(),
        bias=None,
        alibi_slopes=None,
        sm_scale=sm_scale,
        dropout_p=0.0,
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlens_q=config.seqlen_q,
        max_seqlens_k=config.seqlen_kv,
        causal=causal,
        return_scores=False,
        use_exp2=True,
        layout="bhsd",
    )[0]
    o_kernel.backward(do)

    pairs = [
        ("O", o_ref, o_kernel, 0.99, 20),
        ("dQ", q_ref.grad, q_kernel.grad, 0.995, 18),
        ("dK", k_ref.grad, k_kernel.grad, 0.995, 18),
        ("dV", v_ref.grad, v_kernel.grad, 0.995, 18),
    ]
    rows = []
    for name, ref, got, _, _ in pairs:
        rows.append([f"{name} (public autograd vs sdpa)", calc_snr(ref, got), calc_cossim(ref, got)])
    print()
    print(tabulate(rows, headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    for name, ref, got, sim_threshold, snr_threshold in pairs:
        sim = calc_cossim(ref, got)
        snr = calc_snr(ref, got)
        assert sim > sim_threshold, f"{name} public-autograd-vs-sdpa cosine-sim too low: {sim}"
        assert snr > snr_threshold, f"{name} public-autograd-vs-sdpa SNR too low: {snr}"


# ---------------------------------------------------------------------------
# Backward Layer 2 — kernel vs Stage-2 (A1) golden reference
#   The A1 backward kernel runs tl.dot_scaled in-kernel, so this requires CDNA4
#   (like the forward Layer 2). Calls the backward op directly with the
#   forward-reference o/lse (bypassing the forward kernel) and compares to the
#   Stage-2 reference (same full quantization both sides -> gap = port bugs).
# ---------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("config", test_cases)
@pytest.mark.parametrize("causal", [True])
def test_backward_kernel_matches_reference(config, causal):
    """A1 backward kernel vs Stage-2 golden reference — isolates port bugs. CDNA4-only."""
    from alto.kernels.mxfp8.mxfp8_quantization import convert_to_mxfp8

    device = "cuda"
    dtype = torch.bfloat16
    q, k, v = _make_qkv_bhsd(4, config, device, dtype)
    do = _make_do_bhsd(4, config, device, dtype)
    sm_scale = config.head_dim_qk**(-0.5)

    o_ref, lse_ref = mxfp8_attention_forward_reference(q, k, v, sm_scale, causal)

    # Quantize operands the way the forward autograd does, then run the backward op.
    q8, q_scale = convert_to_mxfp8(q, mxfp_format="e4m3", axis=-1, is_2d_block=True)
    k8, k_scale = convert_to_mxfp8(k, mxfp_format="e4m3", axis=-1, is_2d_block=True)
    v8, v_scale = convert_to_mxfp8(v, mxfp_format="e4m3", axis=-1, is_2d_block=True)

    dq_k, dk_k, dv_k = torch.ops.alto.attention_mxfp8_backward_triton_impl(
        do.contiguous(),
        q8.contiguous(),
        k8.contiguous(),
        v8.contiguous(),
        o_ref.contiguous(),
        lse_ref.contiguous(),
        q_scale,
        k_scale,
        v_scale,
        sm_scale=sm_scale,
        causal=causal,
        layout="bhsd",
        cu_seqlens_q=0,
        cu_seqlens_k=0,
        max_seqlen_q=config.seqlen_q,
        max_seqlen_k=config.seqlen_kv,
        use_exp2=True,
    )
    dq_r, dk_r, dv_r = mxfp8_attention_backward_reference_stage2(q, k, v, do, o_ref, lse_ref, sm_scale, causal)

    rows = []
    pairs = [("dQ", dq_r, dq_k), ("dK", dk_r, dk_k), ("dV", dv_r, dv_k)]
    for name, ref, got in pairs:
        rows.append([f"{name} (kernel vs golden)", calc_snr(ref, got), calc_cossim(ref, got)])
    print()
    print(tabulate(rows, headers=["Tensor", "SNR", "Cosine Sim"], tablefmt="github"))

    # Threshold placeholder — calibrate on the first CDNA4 run.
    for name, ref, got in pairs:
        sim = calc_cossim(ref, got)
        snr = calc_snr(ref, got)
        assert sim > 0.99, f"{name} kernel-vs-golden cosine-sim too low: {sim}"
        assert snr > 25, f"{name} kernel-vs-golden SNR too low: {snr}"
