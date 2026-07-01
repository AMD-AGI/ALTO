# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Tests for packed MX9 quantization (convert_to_mx9 / convert_from_mx9).

Acceptance criterion: unpack(pack(x)) == mx9_clamp127_ref(x) bit-exact.
clamp127_ref is the clamp-127 variant of mx9_fake_quantize (demoted +/-128
truncated to +/-127), serving as the true reference for the pack path.

Tests are organized in four layers:
  1. Storage format assertions: dtype / shape correctness.
  2. Round-trip bit-exact: vs clamp127_ref, covering shape/dtype/axis.
  3. Boundary & properties: zeros, Inf, block independence, idempotency,
     padding, +/-128 edge.
  4. Invalid input rejection.
"""

import pytest
import torch
import torch.nn.functional as F

from alto.modifiers.quantization import mx as _mxref
from alto.modifiers.quantization.mx import mx9_fake_quantize, BLOCK_SIZE
from alto.kernels.mx.mx9_quantization import convert_to_mx9, convert_from_mx9
from alto.kernels.fp4.testing_utils import calc_snr

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA/HIP device"
)

# --------------------------------------------------------------------------- #
# Reference implementation: clamp-127 variant of mx9_fake_quantize
# --------------------------------------------------------------------------- #

def _bit_exponent(t: torch.Tensor) -> torch.Tensor:
    """Extract exponent field from float bits with bias removal (approx
    floor(log2|t|)), branching by dtype.

    Aligned with the kernel's ``_floor_exp`` / mx.py's
    ``_exponent_frexp_no_exception``, but independently inlined here (does not
    call mx.py) to keep the two reference implementations independent.
    Must operate on *native dtype* (fp16 bias=15, bf16/fp32 bias=127); callers
    must NOT pre-cast to .float(). NaN/Inf are zeroed before extraction to
    prevent pollution. ``& mask`` clears sign-extension bits from arithmetic
    right shift, so negative values are safe. Returns long.
    """
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if t.dtype == torch.float32:
        return (((t.view(torch.int32) >> 23) & 0xFF) - 127).long()
    if t.dtype == torch.bfloat16:
        return (((t.view(torch.int16) >> 7) & 0xFF) - 127).long()
    if t.dtype == torch.float16:
        return (((t.view(torch.int16) >> 10) & 0x1F) - 15).long()
    raise ValueError(f"unsupported dtype {t.dtype}")


def _mx9_clamp127_ref(x: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Clamp-127 variant of mx9_fake_quantize, serving as the pack path reference.

    Only differs from mx9_fake_quantize in that demoted elements' quant_max is
    also truncated to 127 (matching the pack path's int8 clamp ceiling). The two
    diverge only for the extremely rare "demoted AND round reaches +/-128" edge.

    Exponent extraction uses bit-field extraction (``_bit_exponent``, on native
    dtype), aligned with kernel / mx.py -- an earlier version using
    ``floor(log2())`` had fp32 rounding errors near octave boundaries, causing
    occasional 1-octave divergence in small-magnitude regions; now eliminated.
    """
    orig_dtype = x.dtype
    orig_shape = x.shape
    last = orig_shape[-1]

    # Keep native dtype for blocking (exponent extraction requires native dtype).
    pad = (block_size - last % block_size) % block_size
    x2d = x.reshape(-1, last)
    if pad:
        x2d = F.pad(x2d, (0, pad))
    rows, cols = x2d.shape
    x3d = x2d.reshape(rows, cols // block_size, block_size)   # native dtype

    # Block shared exponent: amax(native dtype) -> bit extract exponent.
    clean = torch.nan_to_num(x3d, nan=0.0, posinf=0.0, neginf=0.0)
    amax = clean.abs().amax(dim=-1, keepdim=True)            # [rows, nblk, 1] native dtype
    max_exp = _bit_exponent(amax)                            # long

    # Per-element exponent + demote flag.
    t_exp = _bit_exponent(x3d)                               # long
    demote_elem = (max_exp - t_exp) >= 1

    d = demote_elem.reshape(rows, cols // block_size, block_size // 2, 2)
    pair = (d.sum(dim=-1, keepdim=True) == 2)
    pair = pair.expand_as(d).reshape(rows, cols // block_size, block_size)

    shared_exp = max_exp - pair.long()
    # Sync with kernel: scale exponent clamped to minimum normal -126.
    scale_exp = torch.clamp(shared_exp - 8 + 2, min=-126)    # quant_bit=8
    scale = torch.exp2(scale_exp.float())

    q = torch.round(x3d.float() / scale).clamp(-127, 127)    # clamp-127 variant
    out3d = q * scale

    out2d = out3d.reshape(rows, cols)
    if pad:
        out2d = out2d[:, :last]
    return out2d.reshape(orig_shape).to(orig_dtype)


def _mx9_clamp127_ref_via_mxpy(x: torch.Tensor, block_size: int = BLOCK_SIZE,
                               quant_bit: int = 8) -> torch.Tensor:
    """Second independent reference: reuses mx.py's real primitives (_t_exponent /
    _reshape_to_blocks / SHARED_PRIME_BIT_GROUP), only clamping the quantized
    value to +/-127.

    Computes the same result as `_mx9_clamp127_ref` above (pure handwritten
    reimplementation), but **reuses production code mx.py's building blocks**.
    Cross-checking two independent references prevents "kernel and one reference
    share the same bug" from going undetected.
    """
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
    # Sync with kernel: scale exponent clamped to minimum normal -126 (mxfp4 style).
    scale_exp = torch.clamp(shared_exp - quant_bit + 2, min=-126)
    scale = torch.pow(2.0, scale_exp)
    out = torch.round(inp / scale)
    out = torch.clamp(out, -127, 127) * scale          # clamp-127 variant
    out = out.reshape(out.size(0), -1)
    out = out[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return out.transpose(axis, -1)


def _rand(shape, dtype, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dtype)


def _rand_with_outliers(shape, dtype, seed=0, p=0.005, spike=100.0):
    """Random data + sparse outlier spikes (ported from mxfp4 tests' prepare_data).

    ~p fraction of elements get a spike* addition, creating heavy-tailed blocks
    that stress demote/prime/max_exp: one huge value in a block pulls the shared
    scale up, squashing the remaining small values. Pure randn never hits this.
    """
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype)
    mask = torch.bernoulli(torch.full_like(x, p))
    x = x + spike * torch.randn_like(x) * mask
    return x


def _roundtrip(x, block_size=BLOCK_SIZE, axis=-1):
    q, max_exp, prime = convert_to_mx9(x, block_size=block_size, axis=axis)
    return convert_from_mx9(q, max_exp, prime, x.dtype, x.shape,
                            block_size=block_size, axis=axis)


# --------------------------------------------------------------------------- #
# Layer 1: Storage format assertions
# --------------------------------------------------------------------------- #

def test_output_dtypes_and_shapes():
    x = _rand((4, 64), torch.float32).cuda()
    q, max_exp, prime = convert_to_mx9(x, block_size=16)
    n_blocks = (4 * 64) // 16
    assert q.dtype == torch.int8
    assert max_exp.dtype == torch.uint8
    assert prime.dtype == torch.uint8
    assert q.shape == (n_blocks, 16)
    assert max_exp.shape == (n_blocks,)
    assert prime.shape == (n_blocks, 1)      # block_size=16 -> 8 pairs -> 1 byte


def test_max_exp_biased_range():
    # E8M0 stores true exponent + 127; finite fp32 true exponent in [-126, 127],
    # biased to [1, 254]; should never be 0 or 255.
    x = _rand((16, 64), torch.float32).cuda()
    _, max_exp, _ = convert_to_mx9(x, block_size=16)
    assert max_exp.min().item() >= 1
    assert max_exp.max().item() <= 254


def test_q_within_int8_range():
    x = _rand((8, 128), torch.float32).cuda()
    q, _, _ = convert_to_mx9(x, block_size=16)
    assert q.min().item() >= -127
    assert q.max().item() <= 127


# --------------------------------------------------------------------------- #
# Layer 1b: Per-component intermediate value verification
#   (absorbed from _mx9_intermediate_check.py scaffold)
#   Round-trip only verifies "combined result is correct" which can mask
#   individual component errors; here we compare convert_to_mx9's q / max_exp /
#   prime individually against torch-recomputed intermediates.
# --------------------------------------------------------------------------- #

def _torch_intermediates(x, block_size=BLOCK_SIZE, quant_bit=8):
    """Pure PyTorch recomputation of the three-part tuple (q_int8, max_exp_u8,
    prime_u8, pair)."""
    last = x.shape[-1]
    x2d = x.contiguous().reshape(-1, last)
    pad = (block_size - last % block_size) % block_size
    if pad:
        x2d = F.pad(x2d, (0, pad))
    rows, cols = x2d.shape
    n_blocks = rows * (cols // block_size)
    xb = x2d.reshape(n_blocks, block_size)

    clean = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
    amax = clean.abs().amax(dim=1, keepdim=True)
    max_exp = _bit_exponent(amax)
    t_exp = _bit_exponent(xb)
    demote = (max_exp - t_exp) >= 1

    n_pairs = block_size // 2
    d = demote.reshape(n_blocks, n_pairs, 2)
    pair = (d.sum(-1) == 2).to(torch.int64)
    pair_b = pair.reshape(n_blocks, n_pairs, 1).expand(n_blocks, n_pairs, 2).reshape(n_blocks, block_size)
    shared_exp = max_exp - pair_b

    scale_exp = torch.clamp(shared_exp - quant_bit + 2, min=-126)
    scale = torch.pow(2.0, scale_exp.to(torch.float32))
    q = torch.round(xb.to(torch.float32) / scale)
    q = torch.clamp(q, -127, 127).to(torch.int8)

    max_exp_u8 = (max_exp.reshape(n_blocks) + 127).to(torch.uint8)

    n_prime_bytes = n_pairs // 8
    weights = (2 ** torch.arange(8, device=x.device)).to(torch.int64)
    pb = pair.reshape(n_blocks, n_prime_bytes, 8)
    prime_u8 = (pb * weights.view(1, 1, 8)).sum(-1).to(torch.uint8)
    return q, max_exp_u8, prime_u8, pair


@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (3, 40), (2, 3, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_intermediates_match_torch_reference(shape, dtype):
    """Each component (q, max_exp, prime) matches torch recomputation bit-exact."""
    x = _rand(shape, dtype).cuda()
    q, max_exp, prime = convert_to_mx9(x, block_size=BLOCK_SIZE)
    rq, re, rp, _ = _torch_intermediates(x, block_size=BLOCK_SIZE)
    assert torch.equal(q, rq), f"q mismatch: {(q != rq).sum().item()} elements"
    assert torch.equal(max_exp, re), f"max_exp mismatch"
    assert torch.equal(prime, rp), f"prime mismatch"


def test_intermediates_constructed_demote():
    """Constructed case: verify deterministic prime bitmap behavior.

    blk = [4, 1,1,...,1]: 4 has exp=2 (max); 1 has exp=0 (demoted).
    pair0=(4,1) only one demoted -> prime=0; pair1..7=(1,1) both demoted -> prime=1.
    """
    x = torch.ones((1, BLOCK_SIZE), dtype=torch.float32).cuda()
    x[0, 0] = 4.0
    _, _, _, pair = _torch_intermediates(x)
    assert pair[0].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    q, max_exp, prime = convert_to_mx9(x)
    rq, re, rp, _ = _torch_intermediates(x)
    assert torch.equal(q, rq)
    assert torch.equal(max_exp, re)
    assert torch.equal(prime, rp)


# --------------------------------------------------------------------------- #
# Layer 2: Round-trip bit-exact vs clamp127_ref
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", [(4, 64), (8, 16), (2, 3, 32), (3, 40), (1, 4096), (128, 4095)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("block_size", [16])
def test_roundtrip_matches_clamp127_ref(shape, dtype, block_size):
    x = _rand(shape, dtype).cuda()
    ref = _mx9_clamp127_ref(x, block_size=block_size)
    out = _roundtrip(x, block_size=block_size)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert torch.equal(out, ref), \
        f"max diff={( out.float() - ref.float()).abs().max().item()}"


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_roundtrip_axis(axis):
    x = _rand((32, 64), torch.float32).cuda()
    out = _roundtrip(x, block_size=16, axis=axis)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    ref = _mx9_clamp127_ref(x.transpose(axis, -1).contiguous(),
                            block_size=16).transpose(axis, -1).contiguous()
    assert torch.equal(out, ref), \
        f"axis={axis} max diff={(out - ref).abs().max().item()}"


@pytest.mark.parametrize("blocks_per_program", [1, 16, 64, 256])
def test_blocks_per_program_does_not_change_numerics(blocks_per_program):
    x = _rand((32, 512), torch.float32).cuda()
    ref = _roundtrip(x, block_size=BLOCK_SIZE)
    q, me, pr = convert_to_mx9(x, block_size=BLOCK_SIZE,
                                blocks_per_program=blocks_per_program)
    out = convert_from_mx9(q, me, pr, x.dtype, x.shape,
                           block_size=BLOCK_SIZE,
                           blocks_per_program=blocks_per_program)
    assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# Layer 3: Boundary & properties
# --------------------------------------------------------------------------- #

def test_zeros_stay_zero():
    x = torch.zeros((4, 64), dtype=torch.float32).cuda()
    out = _roundtrip(x)
    assert torch.equal(out, x)


def test_inf_clamped_to_finite():
    x = torch.full((1, BLOCK_SIZE), 4.0).cuda()
    x[0, 0] = float("inf")
    x[0, 1] = float("-inf")
    out = _roundtrip(x)
    assert not torch.isinf(out).any()
    ref = _mx9_clamp127_ref(x)
    assert torch.equal(out[0, 2:], ref[0, 2:])


def test_block_independence():
    big   = torch.full((1, BLOCK_SIZE), 100.0).cuda()
    small = torch.full((1, BLOCK_SIZE), 0.01).cuda()
    joint = _roundtrip(torch.cat([big, small], dim=0))
    alone = _roundtrip(small)
    assert torch.equal(joint[1:2], alone)


def test_qdq_idempotent():
    # Output of pack->unpack already lies on the MX9 grid; a second round-trip
    # should produce identical results.
    x = _rand((8, 64), torch.float32).cuda()
    once  = _roundtrip(x)
    twice = _roundtrip(once)
    assert torch.equal(once, twice)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_non_contiguous_input(dtype):
    """Non-contiguous memory input (transposed stride) produces the same result
    as the equivalent contiguous tensor.

    Host-side data_hp.contiguous() handles the re-layout; the kernel always
    receives contiguous memory. This test verifies that path is not skipped.
    """
    x = _rand((64, 32), dtype).cuda()
    x_t = x.T                                  # shape (32, 64), non-contiguous
    assert not x_t.is_contiguous()
    out_t = _roundtrip(x_t, axis=0)
    ref_t = _mx9_clamp127_ref(x, block_size=BLOCK_SIZE).T.contiguous()
    assert torch.equal(out_t, ref_t), \
        f"non-contiguous max diff={(out_t.float()-ref_t.float()).abs().max().item()}"


def test_padding_non_divisible_last_dim():
    x = _rand((3, 40), torch.float32).cuda()   # 40 not divisible by 16
    out = _roundtrip(x, block_size=16)
    ref = _mx9_clamp127_ref(x, block_size=16)
    assert out.shape == x.shape
    assert torch.equal(out, ref)


def test_demote_boundary_clamp127():
    """Construct a demoted element that rounds exactly to +/-128, verify it is
    clamped to +/-127.

    Setup: max_exp=7 (amax in [128,256)), demote scale = 2^(7-7)=1.
    x=127.6 -> round(127.6/1)=128 -> clamp127 truncates to 127 -> reconstructs 127*1=127.
    """
    x = torch.zeros((1, BLOCK_SIZE), dtype=torch.float32).cuda()
    x[0, 0] = 130.0          # amax, max_exp=7, non-demote
    x[0, 1] = 128.0          # pair[0] neighbor (non-demote, ensures amax unchanged)
    x[0, 2] = 127.6          # demote pair[1,2]
    x[0, 3] = 1.0            # demote pair[1,2] neighbor, both demotable
    out = _roundtrip(x)
    ref = _mx9_clamp127_ref(x)
    assert torch.equal(out, ref)
    assert out[0, 2].item() == pytest.approx(127.0)


def test_prime_bitmap_round_trips():
    """Verify prime bitmap pack/unpack symmetry.

    Construct a block: first half (pairs 0-3) all demotable, second half
    (pairs 4-7) not demotable. Expected prime byte: low 4 bits = 0xF,
    high 4 bits = 0x0, i.e. 0x0F.
    Demotable condition: max_exp - t_exp >= 1, i.e. element is at least 1
    octave below amax.
    amax=8 (max_exp=3), first 8 elements=1.0 (t_exp=0, diff 3 >= 1, demotable);
    last 8 elements=8.0 (t_exp=3, diff 0, not demotable).
    """
    x = torch.zeros((1, BLOCK_SIZE), dtype=torch.float32).cuda()
    x[0, :8]  = 1.0    # t_exp=0, max_exp=3, diff 3 -> demotable
    x[0, 8:]  = 8.0    # t_exp=3, max_exp=3, diff 0 -> not demotable, amax
    q, max_exp, prime = convert_to_mx9(x, block_size=BLOCK_SIZE)
    # pairs 0-3 (idx 0-7) all demote -> bits 0-3 = 1; pairs 4-7 not demote -> bits 4-7 = 0
    assert prime[0, 0].item() == 0x0F, f"got {hex(prime[0, 0].item())}"
    out = convert_from_mx9(q, max_exp, prime, x.dtype, x.shape)
    ref = _mx9_clamp127_ref(x)
    assert torch.equal(out, ref)


# --------------------------------------------------------------------------- #
# Layer 3b: Subnormal / tiny-value degenerate blocks
#   (absorbed from _mx9_edge_check.py scaffold)
#   Numerical kernels are most likely to break on 0/0, FTZ, and scale underflow;
#   random data never reaches these cases.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_all_zero_blocks(dtype):
    x = torch.zeros((4, 64), dtype=dtype).cuda()
    out = _roundtrip(x)
    assert torch.equal(out, _mx9_clamp127_ref(x))


def test_one_zero_block_among_normal():
    x = _rand((4, 64), torch.float32).cuda()
    x[1, 16:32] = 0.0          # row 1, second 16-block is all zeros
    out = _roundtrip(x)
    assert torch.equal(out, _mx9_clamp127_ref(x))


@pytest.mark.parametrize("scale_pow", [-120, -135, -140])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tiny_and_subnormal_blocks(scale_pow, dtype):
    # -120: near scale clamp boundary (-126) normal small; -135/-140: subnormal.
    # Verify scale clamped to minimum normal produces no 0/0, no FTZ dependence,
    # and matches reference bit-exact.
    x = (_rand((4, 64), dtype) * (2.0 ** scale_pow)).cuda()
    out = _roundtrip(x)
    ref = _mx9_clamp127_ref(x)
    assert torch.equal(out, ref)
    assert not torch.isnan(out).any()


# --------------------------------------------------------------------------- #
# Layer 3c: Dual independent reference cross-validation
#   (absorbed from _mx9_roundtrip_check.py scaffold)
#   Asserts "two independent references agree with each other" AND "kernel
#   agrees with both", preventing shared bugs from going undetected.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", [(4, 64), (3, 40), (2, 3, 32)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_independent_refs_agree_and_match_kernel(shape, dtype):
    x = _rand(shape, dtype).cuda()
    ref_handwritten = _mx9_clamp127_ref(x)          # pure handwritten reimplementation
    ref_via_mxpy = _mx9_clamp127_ref_via_mxpy(x)    # reuses mx.py production primitives
    # (1) Two independent references must agree (otherwise one has a bug).
    assert torch.equal(ref_handwritten, ref_via_mxpy)
    # (2) Kernel round-trip must match both references.
    out = _roundtrip(x)
    assert torch.equal(out, ref_via_mxpy)


def test_divergence_vs_mxpy_255clamp_is_tiny():
    """Verify that the "intentional divergence vs mx.py original (255-clamp)" is
    tiny, confirming it only occurs at the rare +/-128 boundary.

    Divergence should be far less than total element count (only elements that
    are both demoted AND have quantized value reaching +/-128).
    """
    x = _rand((8, 256), torch.float32).cuda()
    out = _roundtrip(x)
    mxpy = mx9_fake_quantize(x)
    n_div = (out.float() != mxpy.float()).sum().item()
    assert n_div < x.numel() * 0.01, \
        f"divergence {n_div}/{x.numel()} too large, likely a bug rather than +/-128 edge"


# --------------------------------------------------------------------------- #
# Layer 3d: Outlier / heavy-tailed data + quality floor (inspired by mxfp4 tests)
#   Pure randn cannot reach "one huge value in block pulls scale up, squashing
#   remaining small values"; outlier spikes specifically stress demote/prime/
#   max_exp. SNR floor is a human-readable sanity check ("is quantization
#   overall usable?") complementing bit-exact assertions: a completely broken
#   kernel would yield near-0 or negative dB.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape", [(8, 256), (128, 512), (2, 3, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_matches_ref_with_outliers(shape, dtype):
    """Round-trip with heavy-tailed/outlier data still matches reference bit-exact."""
    x = _rand_with_outliers(shape, dtype).cuda()
    ref = _mx9_clamp127_ref(x)
    out = _roundtrip(x)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert torch.equal(out, ref), \
        f"max diff={(out.float() - ref.float()).abs().max().item()}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_snr_floor_clean(dtype):
    """Clean random data round-trip SNR should be well above a conservative floor.

    Floor is set conservatively to catch total failures only, not as a tight bound
    (8-bit/block quantization on clean data typically yields 40+ dB).
    """
    x = _rand((64, 512), dtype).cuda()
    out = _roundtrip(x)
    snr = calc_snr(x, out)
    assert snr > 25.0, f"SNR={snr:.1f} dB too low, quantization likely broken"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_roundtrip_snr_floor_outliers(dtype):
    """With outlier spikes, SNR will drop (small values in outlier blocks get
    squashed), but should still exceed a more conservative floor."""
    x = _rand_with_outliers((64, 512), dtype).cuda()
    out = _roundtrip(x)
    snr = calc_snr(x, out)
    assert snr > 15.0, f"SNR={snr:.1f} dB too low, quantization likely broken"


# --------------------------------------------------------------------------- #
# Layer 4: Invalid input rejection
# --------------------------------------------------------------------------- #

def test_rejects_non_float_dtype():
    x = torch.randint(0, 10, (4, 16)).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx9(x)


def test_rejects_float16():
    x = _rand((4, 16), torch.float16).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx9(x)


def test_rejects_block_size_not_16():
    x = _rand((4, 64), torch.float32).cuda()
    with pytest.raises(AssertionError):
        convert_to_mx9(x, block_size=24)   # not 16
    with pytest.raises(AssertionError):
        convert_to_mx9(x, block_size=32)   # even though multiple of 16, only 16 is supported


def test_rejects_unknown_out_dtype():
    x = _rand((4, 16), torch.float32).cuda()
    q, me, pr = convert_to_mx9(x)
    with pytest.raises(AssertionError):
        convert_from_mx9(q, me, pr, torch.int32, x.shape)
