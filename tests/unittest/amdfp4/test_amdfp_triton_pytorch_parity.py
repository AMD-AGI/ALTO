# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Triton vs. PyTorch parity tests for the UE5M3 quant primitive (AMD-FP4 inner scale).

The PyTorch reference in :mod:`alto.kernels.fp4.fp4_primitives.ue5m3_ops` is
the oracle.  This module verifies that :func:`triton_quantize_to_ue5m3`
produces **bit-identical fp32 snapped values** (with matching NaN masks
under D2') for:

* all 256 UE5M3 grid points (including 0xFF -> NaN),
* a large random fp32 tensor (1M elems),
* eight pathological input patterns drawn from the AMD-FP4 stress matrix
  (zeros, large normals, near-MIN_NORMAL, near-MAX, NaN/Inf, negatives,
  hot channels, mixed magnitudes),
* an explicit NaN/Inf parity sweep.

CUDA / ROCm is required.  The whole module is skipped otherwise.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Triton parity tests require a CUDA / ROCm device",
)

from alto.kernels.fp4.fp4_common import (  # noqa: E402
    UE5M3_EPS,
    UE5M3_MAX,
    UE5M3_MIN_NORMAL,
    UE5M3_NAN_CODE,
    f32_to_ue5m3_uint8,
    quantize_to_ue5m3,
    triton_quantize_to_ue5m3,
    ue5m3_uint8_to_f32,
)


def _assert_nan_aware_equal(a: torch.Tensor, b: torch.Tensor, ctx: str) -> None:
    """Bit-equal comparison that treats two NaN positions as equal.

    Required under D2': both PyTorch and Triton paths emit NaN for the
    spec-aligned NaN code 0xFF, but ``torch.equal`` returns False on NaN
    elements.  We therefore separately check the NaN mask and the finite
    bit-equality.
    """
    nan_a = torch.isnan(a)
    nan_b = torch.isnan(b)
    assert torch.equal(nan_a, nan_b), (
        f"{ctx}: NaN masks differ; first diff idx="
        f"{int((nan_a ^ nan_b).nonzero()[0].item())}"
    )
    finite = ~nan_a
    assert torch.equal(a[finite], b[finite]), (
        f"{ctx}: finite snap values differ; "
        f"max abs diff={(a[finite] - b[finite]).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Parity on the 256-code grid
# ---------------------------------------------------------------------------

@cuda_required
def test_triton_parity_on_grid():
    codes = torch.arange(256, dtype=torch.uint8)
    values = ue5m3_uint8_to_f32(codes).cuda()
    snapped_triton = triton_quantize_to_ue5m3(values).cpu()
    snapped_torch = quantize_to_ue5m3(values.cpu())
    _assert_nan_aware_equal(
        snapped_triton, snapped_torch,
        ctx="Triton vs PyTorch grid-point parity",
    )


@cuda_required
def test_triton_parity_re_encodes_to_same_uint8_on_grid():
    """A stronger statement: the FP32 Triton output, when re-encoded by
    the PyTorch oracle, must produce the same UE5M3 code as the input
    (including 0xFF -> NaN -> 0xFF under D2')."""
    codes = torch.arange(256, dtype=torch.uint8)
    values = ue5m3_uint8_to_f32(codes).cuda()
    snapped = triton_quantize_to_ue5m3(values).cpu()
    re_encoded = f32_to_ue5m3_uint8(snapped)
    assert torch.equal(codes, re_encoded), (
        "Triton snapped values do not re-encode to their original UE5M3 codes; "
        f"first diff at code 0x{int((codes != re_encoded).nonzero()[0].item()):02X}"
    )


# ---------------------------------------------------------------------------
# Parity on dense random data
# ---------------------------------------------------------------------------

@cuda_required
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_triton_parity_random_1m(seed):
    torch.manual_seed(seed)
    # log-uniform magnitudes across the UE5M3 dynamic range
    log_lo = math.log(UE5M3_EPS * 0.1)
    log_hi = math.log(UE5M3_MAX * 4.0)        # 4x oversampling to stress saturation
    x_cpu = torch.empty(1024 * 1024, dtype=torch.float32).uniform_(log_lo, log_hi).exp()
    # Inject occasional negatives / specials so the saturation branches fire.
    n = x_cpu.numel()
    x_cpu[torch.randperm(n)[: n // 64]] *= -1
    specials = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, UE5M3_EPS, UE5M3_MAX],
        dtype=torch.float32,
    )
    x_cpu[: specials.numel()] = specials

    x_gpu = x_cpu.cuda()
    snapped_triton = triton_quantize_to_ue5m3(x_gpu).cpu()
    snapped_torch = quantize_to_ue5m3(x_cpu)
    _assert_nan_aware_equal(
        snapped_triton, snapped_torch,
        ctx=f"Triton vs PyTorch UE5M3 random-1M parity (seed={seed})",
    )


# ---------------------------------------------------------------------------
# Parity on the AMD-FP4 stress pattern matrix
# ---------------------------------------------------------------------------

def _make_pattern(name: str, n: int) -> torch.Tensor:
    """Build one of the named stress patterns used across AMD-FP4 tests."""
    torch.manual_seed(hash(name) & 0xFFFFFFFF)
    if name == "zeros":
        return torch.zeros(n, dtype=torch.float32)
    if name == "small_random":
        return torch.randn(n, dtype=torch.float32) * 0.01
    if name == "large_random":
        return torch.randn(n, dtype=torch.float32) * 1e3
    if name == "near_min_normal":
        return torch.empty(n, dtype=torch.float32).uniform_(
            UE5M3_EPS * 0.1, UE5M3_MIN_NORMAL * 2.0
        )
    if name == "near_max":
        return torch.empty(n, dtype=torch.float32).uniform_(
            UE5M3_MAX * 0.5, UE5M3_MAX * 1.5
        )
    if name == "specials":
        base = torch.randn(n, dtype=torch.float32)
        base[0] = float("nan")
        base[1] = float("inf")
        base[2] = -float("inf")
        base[3] = -1.0
        base[4] = 0.0
        return base
    if name == "hot_channel":
        x = torch.full((n,), 0.1, dtype=torch.float32)
        x[: max(1, n // 64)] = UE5M3_MAX * 0.9
        return x
    if name == "mixed_magnitudes":
        x = torch.empty(n, dtype=torch.float32)
        chunk = n // 4
        x[:chunk] = torch.randn(chunk) * UE5M3_EPS * 4
        x[chunk:2 * chunk] = torch.randn(chunk) * 1.0
        x[2 * chunk:3 * chunk] = torch.randn(chunk) * 100.0
        x[3 * chunk:] = torch.randn(n - 3 * chunk) * UE5M3_MAX * 0.5
        return x
    raise ValueError(f"Unknown pattern: {name}")


_PATTERNS = [
    "zeros",
    "small_random",
    "large_random",
    "near_min_normal",
    "near_max",
    "specials",
    "hot_channel",
    "mixed_magnitudes",
]


@cuda_required
@pytest.mark.parametrize("pattern", _PATTERNS)
def test_triton_parity_patterns(pattern):
    n = 4096
    x_cpu = _make_pattern(pattern, n)
    x_gpu = x_cpu.cuda()
    snapped_triton = triton_quantize_to_ue5m3(x_gpu).cpu()
    snapped_torch = quantize_to_ue5m3(x_cpu)
    _assert_nan_aware_equal(
        snapped_triton, snapped_torch,
        ctx=f"Triton vs PyTorch UE5M3 pattern='{pattern}' parity",
    )


# ---------------------------------------------------------------------------
# D2' explicit NaN / Inf parity
# ---------------------------------------------------------------------------

@cuda_required
def test_triton_pytorch_parity_on_nan_inf():
    """PyTorch and Triton MUST emit identical NaN code 0xFF behaviour
    on the canonical 'specials' inputs (NaN, ±Inf, overflow).
    """
    x_cpu = torch.tensor(
        [float("nan"), float("inf"), -float("inf"),
         UE5M3_MAX, UE5M3_MAX * 2.0, 1e30, -1.0, 0.0],
        dtype=torch.float32,
    )
    snapped_triton = triton_quantize_to_ue5m3(x_cpu.cuda()).cpu()
    snapped_torch = quantize_to_ue5m3(x_cpu)

    # NaN positions must agree.
    assert torch.equal(torch.isnan(snapped_triton), torch.isnan(snapped_torch)), (
        f"NaN masks differ: triton={torch.isnan(snapped_triton).tolist()}, "
        f"torch={torch.isnan(snapped_torch).tolist()}"
    )
    # On finite positions the snap values must match bit-for-bit.
    finite = ~torch.isnan(snapped_torch)
    assert torch.equal(snapped_triton[finite], snapped_torch[finite]), (
        f"Finite snap values differ on specials sweep: "
        f"triton={snapped_triton.tolist()}, torch={snapped_torch.tolist()}"
    )
    # Specifically: NaN/+Inf/-Inf/overflow -> NaN; -1.0/0.0 -> 0.0;
    # UE5M3_MAX -> 114688.0.
    assert torch.isnan(snapped_torch[0]).item()      # NaN
    assert torch.isnan(snapped_torch[1]).item()      # +Inf
    assert torch.isnan(snapped_torch[2]).item()      # -Inf
    assert snapped_torch[3].item() == UE5M3_MAX      # exact MAX -> 0xFE
    assert torch.isnan(snapped_torch[4]).item()      # 2*MAX -> overflow NaN
    assert torch.isnan(snapped_torch[5]).item()      # 1e30 -> overflow NaN
    assert snapped_torch[6].item() == 0.0            # -1.0 -> 0
    assert snapped_torch[7].item() == 0.0            # 0.0 -> 0


# ---------------------------------------------------------------------------
# Sanity: idempotency must hold on the Triton path too
# ---------------------------------------------------------------------------

@cuda_required
def test_triton_idempotent():
    """Triton snap must be idempotent on its own output (NaN-aware under D2')."""
    torch.manual_seed(0)
    x = torch.randn(8192, dtype=torch.float32).abs() * 10.0
    x[0] = UE5M3_MAX * 2.0    # overflow -> NaN under D2'
    x[1] = 0.0
    x[2] = float("nan")        # explicit NaN input
    x_gpu = x.cuda()
    q1 = triton_quantize_to_ue5m3(x_gpu)
    q2 = triton_quantize_to_ue5m3(q1)
    _assert_nan_aware_equal(
        q1.cpu(), q2.cpu(),
        ctx="Triton UE5M3 idempotency",
    )
