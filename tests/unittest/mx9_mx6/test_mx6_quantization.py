# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Unit tests for packed MX6 quantization, built up stage by stage.

Testing strategy
----------------
Stages 1-3 are ``@triton.jit`` *device* helpers (callable only from within a
kernel), so each is exercised through a tiny single-program "probe" kernel that
loads inputs, calls the helper, and stores the result back. Stage 4 exposes real
grid kernels, launched directly via minimal inline launchers. In every case the
GPU result is compared **bit-exact** (``torch.equal``) against an independent
pure-PyTorch recomputation of the same math, over a matrix of shapes / dtypes /
block counts, plus constructed edge cases (zeros, NaN/Inf, prime patterns,
sign bitmaps, the +/-15 clamp boundary, block independence).

Coverage by stage
-----------------
  - Stage 1: ``_floor_exp`` / ``_round_half_even`` /
             ``_pack_bits8`` / ``_unpack_bits8``
  - Stage 2: ``_calculate_mx6_exp`` (shared_exp / max_exp / prime pair bits)
  - Stage 3: ``_pack_mx6`` (Python-export-compatible encoding) and the pack->unpack
    round-trip vs the clamp-15 reference
  - Stage 4: ``_convert_to_mx6_kernel`` / ``_convert_from_mx6_kernel`` (storage
    format, single packed tensor layout, E8M0 bias, blocks-per-program invariance,
    end-to-end round-trip)
  - Stage 5: ``convert_to_mx6`` / ``convert_from_mx6`` host wrappers (axis
    transpose, padding-aware shape restore, non-contiguous input, invalid-input
    rejection), mirroring the MX9 host test suite.
"""

import pytest
import torch
import triton
import triton.language as tl

from alto.kernels.mx._mx_common import (
    _floor_exp,
    _round_half_even,
    _pack_bits8,
    _unpack_bits8,
    _calculate_mx_exp as _calculate_mx6_exp,
)
from alto.kernels.mx.mx6_quantization import (
    BLOCK_SIZE,
    PRIME_GROUP,
    QUANT_BIT,
    BLOCKS_PER_PROG_DEFAULT,
    _pack_mx6,
    _unpack_mx6,
    _convert_to_mx6_kernel,
    _convert_from_mx6_kernel,
    convert_to_mx6,
    convert_from_mx6,
)
from alto.modifiers.quantization import mx as _mxref
from alto.modifiers.quantization.mx import mx6_fake_quantize

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)


# --------------------------------------------------------------------------- #
# Pure-PyTorch references (independent reimplementation)
# --------------------------------------------------------------------------- #
def _rand(shape, dtype, seed=6):
    torch.manual_seed(seed)
    return (torch.randn(*shape, dtype=dtype) * 5.0)


def _bit_exponent(t: torch.Tensor) -> torch.Tensor:
    """Extract exponent field from native float bits (approx floor(log2|t|)),
    branching by dtype. Must run on native dtype (no pre-cast to float)."""
    if t.dtype == torch.float32:
        return (((t.view(torch.int32) >> 23) & 0xFF) - 127).long()
    if t.dtype == torch.bfloat16:
        return (((t.view(torch.int16) >> 7) & 0xFF) - 127).long()
    raise ValueError(f"unsupported dtype {t.dtype}")


def _torch_calc_ref(xb: torch.Tensor, block_size: int, prime_group: int):
    """Reference for _calculate_mx6_exp. xb: [n_blocks, block_size] native dtype.
    Returns (shared_exp[int64], max_exp[int64], pair[int64])."""
    n_blocks = xb.shape[0]
    clean = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
    amax = clean.abs().amax(dim=1, keepdim=True)              # [nb, 1] native dtype
    max_exp = _bit_exponent(amax)                             # [nb, 1]
    t_exp = _bit_exponent(clean)                              # [nb, bs]
    demote = (max_exp - t_exp) >= 1

    n_pairs = block_size // prime_group
    d = demote.reshape(n_blocks, n_pairs, prime_group)
    pair = (d.sum(dim=-1) == prime_group)                     # [nb, n_pairs] bool
    pair_b = pair[:, :, None].expand(n_blocks, n_pairs, prime_group).reshape(n_blocks, block_size)
    shared_exp = max_exp - pair_b.long()
    return shared_exp, max_exp.reshape(n_blocks), pair.long()


# --------------------------------------------------------------------------- #
# Probe kernels (test-only harnesses that invoke the device helpers)
# --------------------------------------------------------------------------- #
@triton.jit
def _probe_floor_exp_kernel(x_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, _floor_exp(tl.load(x_ptr + i)))


@triton.jit
def _probe_round_kernel(y_ptr, o_ptr, N: tl.constexpr):
    i = tl.arange(0, N)
    tl.store(o_ptr + i, _round_half_even(tl.load(y_ptr + i)))


@triton.jit
def _probe_bits8_kernel(
    bits_ptr,
    packed_ptr,
    unpacked_ptr,
    BPP: tl.constexpr,
    N_BYTES: tl.constexpr,
):
    N_BITS: tl.constexpr = N_BYTES * 8
    rows = tl.arange(0, BPP)
    bit_cols = tl.arange(0, N_BITS)
    bit_offs = rows[:, None] * N_BITS + bit_cols[None, :]
    bits = tl.load(bits_ptr + bit_offs)

    packed = _pack_bits8(bits, BPP, N_BYTES)
    byte_cols = tl.arange(0, N_BYTES)
    byte_offs = rows[:, None] * N_BYTES + byte_cols[None, :]
    tl.store(packed_ptr + byte_offs, packed)

    unpacked = _unpack_bits8(packed, BPP, N_BYTES)
    tl.store(unpacked_ptr + bit_offs, unpacked)


@triton.jit
def _probe_calc_kernel(
    x_ptr, se_ptr, me_ptr, pr_ptr,
    BPP: tl.constexpr, BLOCK_SIZE: tl.constexpr, PRIME_GROUP: tl.constexpr,
):
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    rows = tl.arange(0, BPP)
    cols = tl.arange(0, BLOCK_SIZE)
    offs = rows[:, None] * BLOCK_SIZE + cols[None, :]
    x = tl.load(x_ptr + offs)
    shared_exp, max_exp, pair = _calculate_mx6_exp(x, BPP, BLOCK_SIZE, PRIME_GROUP)
    tl.store(se_ptr + offs, shared_exp)
    tl.store(me_ptr + rows, max_exp)
    pcols = tl.arange(0, N_PAIRS)
    tl.store(pr_ptr + rows[:, None] * N_PAIRS + pcols[None, :], pair)


# --------------------------------------------------------------------------- #
# Stage 1: _floor_exp
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_floor_exp_matches_bit_extraction(dtype):
    torch.manual_seed(6)
    # Finite non-zero values only (_floor_exp does not sanitize NaN/Inf/0).
    x = (torch.randn(64, dtype=dtype, device="cuda") * 37.0)
    x = torch.where(x.abs() < 1e-3, torch.full_like(x, 1e-3), x)
    out = torch.empty(64, dtype=torch.int32, device="cuda")
    _probe_floor_exp_kernel[(1,)](x, out, N=64)
    ref = _bit_exponent(x).to(torch.int32)
    assert torch.equal(out, ref)


def test_floor_exp_known_powers_of_two():
    x = torch.tensor([1.0, 2.0, 3.9, 4.0, 0.5, 0.25, 130.0, -8.0],
                     dtype=torch.float32, device="cuda")
    out = torch.empty(8, dtype=torch.int32, device="cuda")
    _probe_floor_exp_kernel[(1,)](x, out, N=8)
    # floor(log2|x|): 1->0, 2->1, 3.9->1, 4->2, 0.5->-1, 0.25->-2, 130->7, -8->3
    ref = torch.tensor([0, 1, 1, 2, -1, -2, 7, 3], dtype=torch.int32, device="cuda")
    assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# Stage 1: _round_half_even
# --------------------------------------------------------------------------- #
def test_round_half_even_ties_and_regular():
    y = torch.tensor(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5,
         0.4, 0.6, -0.4, -0.6, 2.4, 2.6, -2.4, -2.6],
        dtype=torch.float32, device="cuda",
    )
    out = torch.empty_like(y)
    _probe_round_kernel[(1,)](y, out, N=y.numel())
    ref = torch.round(y)   # torch.round is round-half-to-even
    assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# Stage 1: _pack_bits8 / _unpack_bits8
# --------------------------------------------------------------------------- #
def test_bits8_pack_order_and_roundtrip():
    bits = torch.tensor(
        [
            [0] * 8 + [1] * 8,
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    packed = torch.empty((2, 2), dtype=torch.uint8, device="cuda")
    unpacked = torch.empty_like(bits)

    _probe_bits8_kernel[(1,)](bits, packed, unpacked, BPP=2, N_BYTES=2)

    expected_packed = torch.tensor(
        [[0x00, 0xFF], [0x55, 0xAA]],
        dtype=torch.uint8,
        device="cuda",
    )
    assert torch.equal(packed, expected_packed)
    assert torch.equal(unpacked, bits)


# --------------------------------------------------------------------------- #
# Stage 2: _calculate_mx6_exp
# --------------------------------------------------------------------------- #
def _run_calc(xb, block_size=BLOCK_SIZE, prime_group=PRIME_GROUP):
    n_blocks = xb.shape[0]
    n_pairs = block_size // prime_group
    se = torch.empty((n_blocks, block_size), dtype=torch.int32, device="cuda")
    me = torch.empty((n_blocks,), dtype=torch.int32, device="cuda")
    pr = torch.empty((n_blocks, n_pairs), dtype=torch.int32, device="cuda")
    _probe_calc_kernel[(1,)](xb, se, me, pr, BPP=n_blocks,
                             BLOCK_SIZE=block_size, PRIME_GROUP=prime_group)
    return se, me, pr


@pytest.mark.parametrize("n_blocks", [8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_calculate_exp_matches_torch(n_blocks, dtype):
    torch.manual_seed(6)
    xb = torch.randn(n_blocks, BLOCK_SIZE, dtype=dtype, device="cuda") * 5.0
    se, me, pr = _run_calc(xb)
    rse, rme, rpr = _torch_calc_ref(xb, BLOCK_SIZE, PRIME_GROUP)
    assert torch.equal(se.long(), rse), "shared_exp mismatch"
    assert torch.equal(me.long(), rme), "max_exp mismatch"
    assert torch.equal(pr.long(), rpr), "prime pair mismatch"


def test_calculate_exp_constructed_prime_pattern():
    """Deterministic prime bitmap: block = [8]*8 + [1]*8.
    amax=8 (max_exp=3); first 8 elems=8 (t_exp=3, not demoted);
    last 8 elems=1 (t_exp=0, demoted). Pairs 0-3 -> 0, pairs 4-7 -> 1."""
    xb = torch.empty(1, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    xb[0, :8] = 8.0
    xb[0, 8:] = 1.0
    se, me, pr = _run_calc(xb)
    assert me[0].item() == 3
    assert pr[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    # demoted pairs use max_exp - 1 = 2; non-demoted stay at 3.
    assert se[0].tolist() == [3] * 8 + [2] * 8


def test_calculate_exp_all_zero_block():
    """All-zero block: amax=0 -> _floor_exp(0)=-127; nothing demoted."""
    xb = torch.zeros(2, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    se, me, pr = _run_calc(xb)
    assert torch.all(me == -127)
    assert torch.all(pr == 0)
    assert torch.all(se == -127)


def test_calculate_exp_nan_inf_do_not_pollute():
    """NaN/Inf are sanitized out of the exponent statistics: a block of finite
    values plus one Inf must yield the same max_exp as without the Inf."""
    xb = torch.full((1, BLOCK_SIZE), 3.0, dtype=torch.float32, device="cuda")
    xb[0, 0] = float("inf")
    xb[0, 1] = float("nan")
    se, me, pr = _run_calc(xb)
    # Finite amax = 3 -> max_exp = 1 (Inf/NaN sanitized to 0, ignored).
    assert me[0].item() == 1
    rse, rme, rpr = _torch_calc_ref(xb, BLOCK_SIZE, PRIME_GROUP)
    assert torch.equal(me.long(), rme)
    assert torch.equal(pr.long(), rpr)


# --------------------------------------------------------------------------- #
# Stage 3 references: Python-export-compatible pack encoding + clamp-15 QDQ
# --------------------------------------------------------------------------- #
def _torch_q_and_shared_exp(xb, block_size, prime_group, quant_bit):
    """Shared helper: per-element clamp-15 quantized int q and shared_exp."""
    n_blocks = xb.shape[0]
    clean = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
    amax = clean.abs().amax(dim=1, keepdim=True)
    max_exp = _bit_exponent(amax)
    t_exp = _bit_exponent(clean)
    demote = (max_exp - t_exp) >= 1
    n_pairs = block_size // prime_group
    d = demote.reshape(n_blocks, n_pairs, prime_group)
    pair = (d.sum(dim=-1) == prime_group)
    pair_b = pair[:, :, None].expand(n_blocks, n_pairs, prime_group).reshape(n_blocks, block_size)
    shared_exp = max_exp - pair_b.long()
    q_hi = (1 << (quant_bit - 1)) - 1
    scale_exp = torch.clamp(shared_exp - quant_bit + 2, min=-126)
    scale = torch.exp2(scale_exp.float())
    q = torch.round(xb.float() / scale).clamp(-q_hi, q_hi)
    return q, scale, pair.long()


def _torch_pack_ref(xb, block_size, prime_group, quant_bit):
    """Reference Python export byte encoding.

    Returns ``(sign_bytes, mantissa_bytes, prime_bytes, max_exp_biased)``. The
    full block layout is ``[max_exp, prime, sign, mantissa]``.
    """
    n_blocks = xb.shape[0]
    q, _, pair = _torch_q_and_shared_exp(xb, block_size, prime_group, quant_bit)
    qi = q.to(torch.int64)
    mantissa = qi & 0xF
    sign = (qi < 0).to(torch.int64)

    n_sign_bytes = block_size // 8
    w8 = (2 ** torch.arange(8, device=xb.device)).to(torch.int64)
    sign_bytes = (sign.reshape(n_blocks, n_sign_bytes, 8) * w8).sum(-1)

    n_mantissa_bytes = block_size // 2
    nib_w = torch.tensor([1, 16], device=xb.device, dtype=torch.int64)
    mantissa_bytes = (mantissa.reshape(n_blocks, n_mantissa_bytes, 2) * nib_w).sum(-1)

    n_pairs = block_size // prime_group
    n_prime_bytes = n_pairs // 8
    prime_bytes = (pair.reshape(n_blocks, n_prime_bytes, 8) * w8).sum(-1)

    clean = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
    max_exp_biased = _bit_exponent(clean.abs().amax(1, keepdim=True)).reshape(n_blocks) + 127
    return sign_bytes, mantissa_bytes, prime_bytes, max_exp_biased


def _mx6_clamp15_blockref(xb, block_size, prime_group, quant_bit):
    """Clamp-15 QDQ reference at block granularity: xb[nb, bs] -> y[nb, bs] fp32."""
    q, scale, _ = _torch_q_and_shared_exp(xb, block_size, prime_group, quant_bit)
    return q * scale


_TL_DTYPE = {torch.float32: tl.float32, torch.bfloat16: tl.bfloat16}


@triton.jit
def _probe_pack_kernel(
    x_ptr, sign_ptr, mantissa_ptr, prime_ptr,
    BPP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr, QUANT_BIT: tl.constexpr,
):
    N_PAIRS: tl.constexpr = BLOCK_SIZE // PRIME_GROUP
    N_SIGN_BYTES: tl.constexpr = BLOCK_SIZE // 8
    N_MANTISSA_BYTES: tl.constexpr = BLOCK_SIZE // 2
    N_PRIME_BYTES: tl.constexpr = N_PAIRS // 8
    rows = tl.arange(0, BPP)
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + rows[:, None] * BLOCK_SIZE + cols[None, :])
    shared_exp, max_exp, pair = _calculate_mx6_exp(x, BPP, BLOCK_SIZE, PRIME_GROUP)
    sign_b, mantissa_b, prime_b = _pack_mx6(x, shared_exp, pair, BPP, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    scol = tl.arange(0, N_SIGN_BYTES)
    tl.store(sign_ptr + rows[:, None] * N_SIGN_BYTES + scol[None, :], sign_b)
    mcol = tl.arange(0, N_MANTISSA_BYTES)
    tl.store(mantissa_ptr + rows[:, None] * N_MANTISSA_BYTES + mcol[None, :], mantissa_b)
    pcol = tl.arange(0, N_PRIME_BYTES)
    tl.store(prime_ptr + rows[:, None] * N_PRIME_BYTES + pcol[None, :], prime_b)


@triton.jit
def _probe_roundtrip_kernel(
    x_ptr, y_ptr,
    BPP: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    PRIME_GROUP: tl.constexpr, QUANT_BIT: tl.constexpr, OUT_DTYPE: tl.constexpr,
):
    rows = tl.arange(0, BPP)
    cols = tl.arange(0, BLOCK_SIZE)
    offs = rows[:, None] * BLOCK_SIZE + cols[None, :]
    x = tl.load(x_ptr + offs)
    shared_exp, max_exp, pair = _calculate_mx6_exp(x, BPP, BLOCK_SIZE, PRIME_GROUP)
    sign_b, mantissa_b, prime_b = _pack_mx6(x, shared_exp, pair, BPP, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    y = _unpack_mx6(sign_b, mantissa_b, max_exp[:, None], prime_b, OUT_DTYPE,
                    BPP, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    tl.store(y_ptr + offs, y)


def _run_pack(xb):
    n_blocks = xb.shape[0]
    sign = torch.empty((n_blocks, BLOCK_SIZE // 8), dtype=torch.uint8, device="cuda")
    mantissa = torch.empty((n_blocks, BLOCK_SIZE // 2), dtype=torch.uint8, device="cuda")
    prime = torch.empty((n_blocks, (BLOCK_SIZE // PRIME_GROUP) // 8), dtype=torch.uint8, device="cuda")
    _probe_pack_kernel[(1,)](xb, sign, mantissa, prime, BPP=n_blocks, BLOCK_SIZE=BLOCK_SIZE,
                             PRIME_GROUP=PRIME_GROUP, QUANT_BIT=QUANT_BIT)
    return sign, mantissa, prime


def _run_roundtrip(xb):
    n_blocks = xb.shape[0]
    y = torch.empty((n_blocks, BLOCK_SIZE), dtype=xb.dtype, device="cuda")
    _probe_roundtrip_kernel[(1,)](xb, y, BPP=n_blocks, BLOCK_SIZE=BLOCK_SIZE,
                                  PRIME_GROUP=PRIME_GROUP, QUANT_BIT=QUANT_BIT,
                                  OUT_DTYPE=_TL_DTYPE[xb.dtype])
    return y


# --------------------------------------------------------------------------- #
# Stage 3: _pack_mx6 encoding (sign bitmap / low-4-bit mantissa / prime)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_blocks", [8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pack_encoding_matches_torch(n_blocks, dtype):
    torch.manual_seed(6)
    xb = torch.randn(n_blocks, BLOCK_SIZE, dtype=dtype, device="cuda") * 5.0
    sign, mantissa, prime = _run_pack(xb)
    rsign, rmantissa, rprime, _ = _torch_pack_ref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    assert torch.equal(sign.long(), rsign), "sign bitmap mismatch"
    assert torch.equal(mantissa.long(), rmantissa), "mantissa nibble mismatch"
    assert torch.equal(prime.long(), rprime), "prime bitmap mismatch"


def test_pack_mantissa_fits_4_bits():
    """Every mantissa nibble must be <= 15."""
    torch.manual_seed(1)
    xb = torch.randn(16, BLOCK_SIZE, dtype=torch.float32, device="cuda") * 50.0
    _, mantissa, _ = _run_pack(xb)
    lo = mantissa & 0x0F
    hi = (mantissa >> 4) & 0x0F
    assert lo.max().item() <= 15 and hi.max().item() <= 15


def test_pack_mantissa_uses_twos_complement_low4():
    """q=-3 is encoded as sign=1 plus low4=0xD, matching the Python packer."""
    xb = torch.zeros(1, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    xb[0, 0] = 8.0     # max_exp=3, scale=1 for pair0
    xb[0, 1] = -3.0
    sign, mantissa, _ = _run_pack(xb)
    assert sign[0, 0].item() & 0b10
    assert mantissa[0, 0].item() == 0xD8


def test_pack_sign_bitmap_constructed():
    """Alternating signs -> sign byte 0b10101010 = 0xAA (odd elements negative)."""
    xb = torch.ones(1, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    xb[0, 1::2] = -1.0
    sign, _, _ = _run_pack(xb)
    # elements 0..7 -> byte0: bits 1,3,5,7 set = 0xAA; elements 8..15 -> byte1 = 0xAA
    assert sign[0, 0].item() == 0xAA
    assert sign[0, 1].item() == 0xAA


# --------------------------------------------------------------------------- #
# Stage 3: pack -> unpack round-trip (bit-exact vs clamp-15 reference)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_blocks", [8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_matches_clamp15_ref(n_blocks, dtype):
    torch.manual_seed(6)
    xb = torch.randn(n_blocks, BLOCK_SIZE, dtype=dtype, device="cuda") * 5.0
    y = _run_roundtrip(xb)
    ref = _mx6_clamp15_blockref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT).to(dtype)
    assert y.dtype == dtype
    assert torch.equal(y, ref), f"max diff={(y.float() - ref.float()).abs().max().item()}"


def test_roundtrip_zeros_stay_zero():
    xb = torch.zeros(4, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    y = _run_roundtrip(xb)
    assert torch.equal(y, xb)


def test_roundtrip_sign_preserved():
    torch.manual_seed(3)
    xb = torch.randn(8, BLOCK_SIZE, dtype=torch.float32, device="cuda") * 5.0
    y = _run_roundtrip(xb)
    ref = _mx6_clamp15_blockref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    # nonzero reconstructed values keep the reference sign
    nz = ref != 0
    assert torch.equal(torch.sign(y[nz]), torch.sign(ref[nz]))


def test_roundtrip_idempotent():
    """pack->unpack output already lies on the MX6 grid; a second pass is identical."""
    torch.manual_seed(0)
    xb = torch.randn(8, BLOCK_SIZE, dtype=torch.float32, device="cuda") * 5.0
    once = _run_roundtrip(xb)
    twice = _run_roundtrip(once)
    assert torch.equal(once, twice)


def test_roundtrip_demote_boundary_clamp15():
    """Demoted element rounding to +/-16 must be clamped to +/-15.

    max_exp=3 (amax in [8,16)); demote scale = 2^(3-4)=0.5. An element at 7.6 in a
    demotable pair -> round(7.6/0.5)=round(15.2)=15 stays; push to reach 16:
    value 7.9 -> round(15.8)=16 -> clamp 15 -> reconstruct 15*0.5=7.5.
    """
    xb = torch.zeros(1, BLOCK_SIZE, dtype=torch.float32, device="cuda")
    xb[0, 0] = 9.0     # amax -> max_exp=3, non-demote anchor
    xb[0, 1] = 8.0     # pair0 neighbor (non-demote)
    xb[0, 2] = 7.9     # demote pair1
    xb[0, 3] = 0.5     # demote pair1 neighbor -> both demotable
    y = _run_roundtrip(xb)
    ref = _mx6_clamp15_blockref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
    assert torch.equal(y, ref)
    assert y[0, 2].item() == pytest.approx(7.5)


# --------------------------------------------------------------------------- #
# Stage 4: grid kernels (_convert_to/from_mx6_kernel)
#   Minimal inline launchers drive the kernels directly on a contiguous 2D
#   tensor whose last dim is a multiple of block_size (no transpose / padding --
#   those belong to the Stage 5 host wrappers). This isolates the kernel wiring:
#   stride-addressed load/store, single packed tensor layout, E8M0 bias, masking.
# --------------------------------------------------------------------------- #
_N_PRIME_BYTES = (BLOCK_SIZE // PRIME_GROUP) // 8    # 1
_N_SIGN_BYTES = BLOCK_SIZE // 8                      # 2
_N_MANTISSA_BYTES = BLOCK_SIZE // 2                  # 8
_N_PACKED_BYTES = 1 + _N_PRIME_BYTES + _N_SIGN_BYTES + _N_MANTISSA_BYTES  # 12
_PRIME_OFFSET = 1
_SIGN_OFFSET = _PRIME_OFFSET + _N_PRIME_BYTES
_MANTISSA_OFFSET = _SIGN_OFFSET + _N_SIGN_BYTES


def _launch_to(x2d, bpp=BLOCKS_PER_PROG_DEFAULT):
    rows, cols = x2d.shape
    assert cols % BLOCK_SIZE == 0
    blocks_per_row = cols // BLOCK_SIZE
    n_blocks = rows * blocks_per_row
    packed = torch.empty((n_blocks, _N_PACKED_BYTES), dtype=torch.uint8, device="cuda")
    sr, sc = x2d.stride()
    grid = (triton.cdiv(n_blocks, bpp),)
    _convert_to_mx6_kernel[grid](
        x2d, packed, n_blocks, cols, blocks_per_row, sr, sc,
        BLOCK_SIZE=BLOCK_SIZE, BLOCKS_PER_PROG=bpp,
        PRIME_GROUP=PRIME_GROUP, QUANT_BIT=QUANT_BIT,
        N_PACKED_BYTES=_N_PACKED_BYTES,
    )
    return packed, n_blocks


def _launch_to_ragged(x2d, bpp=BLOCKS_PER_PROG_DEFAULT):
    """Like ``_launch_to`` but does NOT require ``cols`` to be a multiple of
    ``BLOCK_SIZE``. Drives the in-kernel ragged-tail path: ``blocks_per_row`` is
    ``cdiv`` and ``last`` = the unpadded column count, so columns >= last are
    masked to 0 by ``_convert_to_mx6_kernel`` (equivalent to the host-side
    zero-padding the Stage 5 wrapper will do explicitly)."""
    rows, cols = x2d.shape
    blocks_per_row = triton.cdiv(cols, BLOCK_SIZE)
    n_blocks = rows * blocks_per_row
    packed = torch.empty((n_blocks, _N_PACKED_BYTES), dtype=torch.uint8, device="cuda")
    sr, sc = x2d.stride()
    grid = (triton.cdiv(n_blocks, bpp),)
    _convert_to_mx6_kernel[grid](
        x2d, packed, n_blocks, cols, blocks_per_row, sr, sc,
        BLOCK_SIZE=BLOCK_SIZE, BLOCKS_PER_PROG=bpp,
        PRIME_GROUP=PRIME_GROUP, QUANT_BIT=QUANT_BIT,
        N_PACKED_BYTES=_N_PACKED_BYTES,
    )
    return packed, n_blocks


def _launch_from(packed, n_blocks, out_dtype, bpp=BLOCKS_PER_PROG_DEFAULT):
    y = torch.empty((n_blocks, BLOCK_SIZE), dtype=out_dtype, device="cuda")
    spb, spc = packed.stride()
    grid = (triton.cdiv(n_blocks, bpp),)
    _convert_from_mx6_kernel[grid](
        packed, y, n_blocks, spb, spc,
        BLOCK_SIZE=BLOCK_SIZE, BLOCKS_PER_PROG=bpp,
        PRIME_GROUP=PRIME_GROUP, QUANT_BIT=QUANT_BIT,
        OUT_DTYPE=_TL_DTYPE[out_dtype],
        N_PACKED_BYTES=_N_PACKED_BYTES,
    )
    return y


def test_kernel_output_dtypes_and_shapes():
    x = _rand((4, 64), torch.float32).cuda()
    packed, n_blocks = _launch_to(x)
    assert n_blocks == (4 * 64) // BLOCK_SIZE
    assert packed.dtype == torch.uint8 and packed.shape == (n_blocks, _N_PACKED_BYTES)


def test_kernel_max_exp_biased_range():
    x = _rand((16, 64), torch.float32).cuda()
    packed, _ = _launch_to(x)
    assert packed[:, 0].min().item() >= 1
    assert packed[:, 0].max().item() <= 254


def test_kernel_packed_bytes_match_pack_ref():
    """packed row must match Python export layout: [max_exp, prime, sign, mantissa]."""
    for dtype in (torch.float32, torch.bfloat16):
        x = _rand((8, 96), dtype).cuda()
        packed, n_blocks = _launch_to(x)
        xb = x.reshape(n_blocks, BLOCK_SIZE)
        rsign, rmantissa, rprime, rme = _torch_pack_ref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT)
        assert torch.equal(packed[:, 0].long(), rme), f"{dtype} max_exp"
        assert torch.equal(packed[:, _PRIME_OFFSET:_SIGN_OFFSET].long(), rprime), f"{dtype} prime"
        assert torch.equal(packed[:, _SIGN_OFFSET:_MANTISSA_OFFSET].long(), rsign), f"{dtype} sign"
        assert torch.equal(packed[:, _MANTISSA_OFFSET:].long(), rmantissa), f"{dtype} mantissa"


@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (16, 128), (32, 512)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_roundtrip_matches_clamp15_ref(shape, dtype):
    x = _rand(shape, dtype).cuda()
    packed, n_blocks = _launch_to(x)
    y = _launch_from(packed, n_blocks, dtype)
    ref = _mx6_clamp15_blockref(x.reshape(n_blocks, BLOCK_SIZE),
                                BLOCK_SIZE, PRIME_GROUP, QUANT_BIT).to(dtype)
    assert y.shape == (n_blocks, BLOCK_SIZE)
    assert torch.equal(y, ref), f"max diff={(y.float() - ref.float()).abs().max().item()}"


# --------------------------------------------------------------------------- #
# Stage 4: ragged tail (cols not a multiple of BLOCK_SIZE)
#   The launcher forces cols % BLOCK_SIZE == 0 everywhere else, so the kernel's
#   `in_range = col_g < last` tail masking + brow/row remapping never runs. These
#   cases exercise it directly and check equivalence to explicit zero-padding.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(4, 70), (3, 17), (8, 100), (1, 1), (5, 33)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_ragged_tail_equals_zero_padding(shape, dtype):
    rows, cols = shape
    x = _rand(shape, dtype).cuda()
    packed, n_blocks = _launch_to_ragged(x)
    y = _launch_from(packed, n_blocks, dtype)

    # In-kernel tail masking (other=0) is equivalent to zero-padding the quant
    # axis up to a multiple of BLOCK_SIZE, then block-quantizing.
    pad = (BLOCK_SIZE - cols % BLOCK_SIZE) % BLOCK_SIZE
    xpad = torch.nn.functional.pad(x, (0, pad))
    xb = xpad.reshape(n_blocks, BLOCK_SIZE)
    ref = _mx6_clamp15_blockref(xb, BLOCK_SIZE, PRIME_GROUP, QUANT_BIT).to(dtype)

    assert n_blocks == rows * triton.cdiv(cols, BLOCK_SIZE)
    assert y.shape == (n_blocks, BLOCK_SIZE)
    assert torch.equal(y, ref), f"max diff={(y.float() - ref.float()).abs().max().item()}"


def test_kernel_ragged_tail_block_is_partly_padded():
    """A row whose width leaves a 1-real-element tail block: that block's amax /
    q must be computed from the single real value only (rest masked to 0)."""
    # cols=17 -> 2 blocks/row; block1 holds real element 16 then 15 masked zeros.
    x = torch.zeros(1, 17, dtype=torch.float32, device="cuda")
    x[0, :16] = 1.0
    x[0, 16] = 6.0            # sole real element of the tail block
    packed, n_blocks = _launch_to_ragged(x)
    y = _launch_from(packed, n_blocks, torch.float32)
    assert n_blocks == 2
    # Tail block: amax=6 -> max_exp=2 -> E8M0 = 129; element 0 reconstructs to 6,
    # the 15 padded slots stay 0.
    assert packed[1, 0].item() == 2 + 127
    assert y[1, 0].item() == pytest.approx(6.0)
    assert torch.equal(y[1, 1:], torch.zeros(15, device="cuda"))


# --------------------------------------------------------------------------- #
# Stage 4: strided / non-contiguous high-precision input
#   The module's whole point is "addressed by stride, no forced .contiguous()".
#   Every other test passes a contiguous tensor; these pass non-contiguous views
#   (non-unit stride_col, and a transposed stride_row=1 layout) and assert the
#   packed bytes + dequant are bit-identical to the contiguous equivalent.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layout", ["column_slice", "transpose"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_strided_input_matches_contiguous(layout, dtype):
    rows, cols = 8, 64
    if layout == "column_slice":
        # Allocate 2x wide, take every other column -> stride (2*cols, 2).
        big = _rand((rows, cols * 2), dtype).cuda()
        x_strided = big[:, ::2]
    else:
        # Transpose a contiguous (cols, rows) tensor -> shape (rows, cols),
        # stride (1, rows): stride_row=1, non-unit stride_col.
        base = _rand((cols, rows), dtype).cuda()
        x_strided = base.t()
    assert not x_strided.is_contiguous()
    assert x_strided.shape == (rows, cols)

    packed_s, nb_s = _launch_to(x_strided)
    y_s = _launch_from(packed_s, nb_s, dtype)

    x_contig = x_strided.contiguous()
    packed_c, nb_c = _launch_to(x_contig)
    y_c = _launch_from(packed_c, nb_c, dtype)

    assert torch.equal(packed_s, packed_c), "packed bytes differ between strided and contiguous"
    assert torch.equal(y_s, y_c), "dequant differs"


@pytest.mark.parametrize("bpp", [1, 16, 64, 256])
def test_kernel_blocks_per_program_invariant(bpp):
    x = _rand((32, 256), torch.float32).cuda()
    packed0, nb = _launch_to(x, bpp=BLOCKS_PER_PROG_DEFAULT)
    y0 = _launch_from(packed0, nb, torch.float32, bpp=BLOCKS_PER_PROG_DEFAULT)
    packed1, nb1 = _launch_to(x, bpp=bpp)
    y1 = _launch_from(packed1, nb1, torch.float32, bpp=bpp)
    assert torch.equal(packed0, packed1)
    assert torch.equal(y0, y1)


def test_kernel_zeros_and_block_independence():
    big = torch.full((1, BLOCK_SIZE), 100.0, device="cuda")
    small = torch.full((1, BLOCK_SIZE), 0.01, device="cuda")
    joint_x = torch.cat([big, small], dim=0)
    packed_j, nbj = _launch_to(joint_x)
    yj = _launch_from(packed_j, nbj, torch.float32)
    packed_a, nba = _launch_to(small)
    ya = _launch_from(packed_a, nba, torch.float32)
    assert torch.equal(yj[1:2], ya)   # small block unaffected by the large one


# --------------------------------------------------------------------------- #
# Stage 5 reference: full-tensor clamp-15 QDQ reusing mx.py primitives
#   Unlike _mx6_clamp15_blockref (block-granularity, no transpose/padding), this
#   operates on the full tensor at axis=-1 (transpose + pad handled internally
#   by mx.py's _reshape_to_blocks), matching what the Stage 5 host wrappers do.
# --------------------------------------------------------------------------- #
def _mx6_clamp15_ref_via_mxpy(x, block_size=BLOCK_SIZE, quant_bit=QUANT_BIT):
    """Second independent clamp-15 reference reusing mx.py production primitives.

    Same math as mx6_fake_quantize but (a) value clamp fixed to +/-15 (matching
    the pack path) and (b) scale exponent clamped to -126 (matching the kernel /
    mxfp4 FTZ-independence). Operates on a full tensor (axis=-1)."""
    axis = -1
    input_dtype = x.dtype
    input_shape = list(x.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    block_x = _mxref._reshape_to_blocks(x.detach(), block_size, axis)
    block_x = torch.nan_to_num(block_x, nan=0.0, posinf=0.0, neginf=0.0)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    max_exp = _mxref._t_exponent(amax)

    inp = _mxref._reshape_to_blocks(x, block_size, axis)
    t_exp = _mxref._t_exponent(inp)
    demote = (max_exp - t_exp) >= 1

    n = _mxref.SHARED_PRIME_BIT_GROUP
    fs = demote.shape
    demote = demote.reshape(*fs[:-1], fs[-1] // n, n)
    demote = torch.sum(demote, -1, keepdim=True) == n
    demote = demote.repeat(*([1] * (demote.dim() - 1)), n).reshape(fs)

    shared_exp = max_exp - demote.long()
    q_hi = (1 << (quant_bit - 1)) - 1
    scale_exp = torch.clamp(shared_exp - quant_bit + 2, min=-126)
    scale = torch.pow(2.0, scale_exp)
    out = torch.round(inp / scale).clamp(-q_hi, q_hi) * scale
    out = out.reshape(out.size(0), -1)
    out = out[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return out.transpose(axis, -1)


# --------------------------------------------------------------------------- #
# Stage 5: host wrappers (convert_to_mx6 / convert_from_mx6)
#   The Stage 4 tests above launch the grid kernels directly on an already-2D,
#   block-aligned tensor. These tests instead go through the public host
#   wrappers, exercising what Stage 4 deliberately skips: axis transpose,
#   arbitrary tensor rank, padding-aware shape restore, and non-contiguous
#   input arising naturally from the transpose (rather than constructed via
#   slicing/``.t()`` as in the Stage 4 strided-input tests).
# --------------------------------------------------------------------------- #
def _host_roundtrip(x, block_size=BLOCK_SIZE, axis=-1):
    packed = convert_to_mx6(x, block_size=block_size, axis=axis)
    return convert_from_mx6(packed, x.dtype, x.shape, block_size=block_size, axis=axis)


def test_host_output_dtype_and_shape():
    x = _rand((4, 64), torch.float32).cuda()
    packed = convert_to_mx6(x, block_size=BLOCK_SIZE)
    n_blocks = (4 * 64) // BLOCK_SIZE
    assert packed.dtype == torch.uint8
    assert packed.shape == (n_blocks, _N_PACKED_BYTES)


def test_host_max_exp_biased_range():
    # E8M0 stores true exponent + 127; finite fp32 true exponent in [-126, 127],
    # biased to [1, 254]; should never be 0 or 255.
    x = _rand((16, 64), torch.float32).cuda()
    packed = convert_to_mx6(x, block_size=BLOCK_SIZE)
    assert packed[:, 0].min().item() >= 1
    assert packed[:, 0].max().item() <= 254


@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (2, 3, 32), (3, 40), (1, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_roundtrip_matches_clamp15_ref(shape, dtype):
    x = _rand(shape, dtype).cuda()
    ref = _mx6_clamp15_ref_via_mxpy(x, block_size=BLOCK_SIZE)
    out = _host_roundtrip(x)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert torch.equal(out, ref), \
        f"max diff={(out.float() - ref.float()).abs().max().item()}"


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_host_roundtrip_axis(axis):
    x = _rand((32, 64), torch.float32).cuda()
    out = _host_roundtrip(x, block_size=BLOCK_SIZE, axis=axis)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    ref = _mx6_clamp15_ref_via_mxpy(
        x.transpose(axis, -1).contiguous(), block_size=BLOCK_SIZE
    ).transpose(axis, -1).contiguous()
    assert torch.equal(out, ref), \
        f"axis={axis} max diff={(out - ref).abs().max().item()}"


@pytest.mark.parametrize("blocks_per_program", [1, 16, 64, 256])
def test_host_blocks_per_program_does_not_change_numerics(blocks_per_program):
    x = _rand((32, 512), torch.float32).cuda()
    ref = _host_roundtrip(x)
    packed = convert_to_mx6(x, block_size=BLOCK_SIZE, blocks_per_program=blocks_per_program)
    out = convert_from_mx6(packed, x.dtype, x.shape, block_size=BLOCK_SIZE,
                           blocks_per_program=blocks_per_program)
    assert torch.equal(out, ref)


def test_host_zeros_stay_zero():
    x = torch.zeros((4, 64), dtype=torch.float32).cuda()
    out = _host_roundtrip(x)
    assert torch.equal(out, x)


def test_host_block_independence():
    big = torch.full((1, BLOCK_SIZE), 100.0).cuda()
    small = torch.full((1, BLOCK_SIZE), 0.01).cuda()
    joint = _host_roundtrip(torch.cat([big, small], dim=0))
    alone = _host_roundtrip(small)
    assert torch.equal(joint[1:2], alone)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_non_contiguous_input(dtype):
    """Non-contiguous memory input (transposed stride) produces the same result
    as the equivalent contiguous tensor: the host's ``transpose(axis, -1)`` plus
    the kernel's stride-addressed load means no ``.contiguous()`` copy is forced."""
    x = _rand((64, 32), dtype).cuda()
    x_t = x.T                                  # shape (32, 64), non-contiguous
    assert not x_t.is_contiguous()
    out_t = _host_roundtrip(x_t, axis=0)
    ref_t = _mx6_clamp15_ref_via_mxpy(x, block_size=BLOCK_SIZE).T.contiguous()
    assert torch.equal(out_t, ref_t), \
        f"non-contiguous max diff={(out_t.float() - ref_t.float()).abs().max().item()}"


def test_host_non_contiguous_packed_input():
    x = _rand((4, 64), torch.float32).cuda()
    packed = convert_to_mx6(x)
    storage = torch.empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=packed.dtype,
        device=packed.device,
    )
    packed_view = storage[:, ::2]
    packed_view.copy_(packed)
    assert not packed_view.is_contiguous()

    expected = convert_from_mx6(packed, x.dtype, x.shape)
    actual = convert_from_mx6(packed_view, x.dtype, x.shape)
    assert torch.equal(actual, expected)


def test_host_padding_non_divisible_last_dim():
    x = _rand((3, 40), torch.float32).cuda()   # 40 not divisible by 16
    out = _host_roundtrip(x, block_size=BLOCK_SIZE)
    ref = _mx6_clamp15_ref_via_mxpy(x, block_size=BLOCK_SIZE)
    assert out.shape == x.shape
    assert torch.equal(out, ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_divergence_vs_mxpy_31clamp_is_tiny(dtype):
    """Host-level counterpart of test_divergence_vs_mxpy_31clamp_is_tiny: the
    +/-15 clamp only diverges from raw mx6_fake_quantize at the rare "demoted
    AND rounds to +/-16" edge."""
    x = _rand((8, 256), dtype).cuda()
    out = _host_roundtrip(x)
    mxpy = mx6_fake_quantize(x, block_size=BLOCK_SIZE)
    n_div = (out.float() != mxpy.float()).sum().item()
    assert n_div < x.numel() * 0.05, \
        f"divergence {n_div}/{x.numel()} too large, likely a bug rather than the +/-16 edge"


# --------------------------------------------------------------------------- #
# Stage 5: invalid input rejection
# --------------------------------------------------------------------------- #
def test_host_rejects_non_float_dtype():
    x = torch.randint(0, 10, (4, 16)).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx6(x)


def test_host_rejects_float16():
    x = _rand((4, 16), torch.float16).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx6(x)


def test_host_rejects_block_size_not_16():
    x = _rand((4, 64), torch.float32).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx6(x, block_size=24)   # not 16
    with pytest.raises(AssertionError):
        convert_to_mx6(x, block_size=32)   # even though multiple of 16, only 16 is supported


@pytest.mark.parametrize("blocks_per_program", [0, 3, -1])
def test_host_rejects_invalid_blocks_per_program(blocks_per_program):
    x = _rand((4, 16), torch.float32).cuda()
    packed = convert_to_mx6(x)
    with pytest.raises(AssertionError, match="positive power of two"):
        convert_to_mx6(x, blocks_per_program=blocks_per_program)
    with pytest.raises(AssertionError, match="positive power of two"):
        convert_from_mx6(
            packed,
            x.dtype,
            x.shape,
            blocks_per_program=blocks_per_program,
        )


def test_host_rejects_empty_quantization_axis():
    x = torch.empty((2, 0), dtype=torch.float32, device="cuda")
    with pytest.raises(AssertionError, match="non-empty quantization axis"):
        convert_to_mx6(x)

    packed = torch.empty((0, _N_PACKED_BYTES), dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError, match="non-empty quantization axis"):
        convert_from_mx6(packed, torch.float32, x.shape)


def test_host_rejects_unknown_out_dtype():
    x = _rand((4, 16), torch.float32).cuda()
    packed = convert_to_mx6(x)
    with pytest.raises(AssertionError):
        convert_from_mx6(packed, torch.int32, x.shape)


def test_host_rejects_bad_packed_dtype():
    x = _rand((4, 16), torch.float32).cuda()
    packed = convert_to_mx6(x).to(torch.int8)   # wrong dtype (should be uint8)
    with pytest.raises(AssertionError):
        convert_from_mx6(packed, torch.float32, x.shape)


def test_host_rejects_out_shape_inconsistent_with_packed():
    x = _rand((4, 64), torch.float32).cuda()
    packed = convert_to_mx6(x)
    with pytest.raises(AssertionError):
        convert_from_mx6(packed, torch.float32, (3, 64))   # wrong row count for n_blocks


@pytest.mark.parametrize("shape", [(4, 64), (8, 96), (16, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_independent_refs_agree_and_match_kernel(shape, dtype):
    x = _rand(shape, dtype).cuda()
    rows, cols = shape
    n_blocks = rows * cols // BLOCK_SIZE
    ref_hand = _mx6_clamp15_blockref(
        x.reshape(n_blocks, BLOCK_SIZE), BLOCK_SIZE, PRIME_GROUP, QUANT_BIT
    ).to(dtype).reshape(rows, cols)
    ref_mxpy = _mx6_clamp15_ref_via_mxpy(x)
    # (1) two independent references must agree bit-exact.
    assert torch.equal(ref_hand, ref_mxpy), "hand ref vs mx.py-based ref diverge"
    # (2) kernel round-trip must match the mx.py-based reference.
    packed, nb = _launch_to(x)
    y = _launch_from(packed, nb, dtype).reshape(rows, cols)
    assert torch.equal(y, ref_mxpy), "kernel vs mx.py-based ref diverge"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_divergence_vs_mxpy_31clamp_is_tiny(dtype):
    """Our +/-15 clamp diverges from raw mx6_fake_quantize (demoted quant_max=31)
    only where a demoted element rounds to +/-16; that fraction must be tiny."""
    x = _rand((32, 256), dtype).cuda()
    rows, cols = x.shape
    packed, nb = _launch_to(x)
    y = _launch_from(packed, nb, dtype).reshape(rows, cols)
    mxpy = mx6_fake_quantize(x, block_size=BLOCK_SIZE)
    n_div = (y.float() != mxpy.float()).sum().item()
    assert n_div < x.numel() * 0.05, (
        f"divergence {n_div}/{x.numel()} too large, likely a bug rather than "
        f"the +/-16 demote edge"
    )
    # Where they differ, mx.py must have the larger magnitude (it kept +/-16..31).
    diff = y.float() != mxpy.float()
    assert torch.all(mxpy.float()[diff].abs() >= y.float()[diff].abs())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
