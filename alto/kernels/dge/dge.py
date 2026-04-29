# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
import triton
import triton.language as tl


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


device = torch.device("cpu")
hp_dtype = torch.float32
k = 5

if is_cdna4():
    BITS = {
        torch.float8_e4m3fn: 8,
        torch.float8_e5m2: 8,
        torch.float4_e2m1fn_x2: 4,
    }
    INT_DTYPES = {
        torch.float8_e4m3fn: torch.uint8,
        torch.float8_e5m2: torch.uint8,
        torch.float4_e2m1fn_x2: torch.uint8,
    }
else:
    BITS = {
        torch.float8_e4m3fnuz: 8,
        torch.float8_e5m2fnuz: 8,
        torch.float4_e2m1fn_x2: 4,
    }
    INT_DTYPES = {
        torch.float8_e4m3fnuz: torch.uint8,
        torch.float8_e5m2fnuz: torch.uint8,
        torch.float4_e2m1fn_x2: torch.uint8,
    }


def calculate_break_points(dtype):
    match dtype:
        case torch.float4_e2m1fn_x2:
            break_points = torch.tensor([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
                                        dtype=hp_dtype,
                                        device=device)
        case _:
            bitwidth = BITS[dtype]
            int_dtype = INT_DTYPES[dtype]
            break_points = torch.arange(0, 2**bitwidth, dtype=int_dtype,
                                        device=device).view(dtype).to(hp_dtype).sort()[0]
            break_points = break_points[torch.isnan(break_points) == False]

    return break_points


BREAK_POINTS = {t: calculate_break_points(t) for t in BITS.keys()}

INTERVALS = {t: (BREAK_POINTS[t][1:] - BREAK_POINTS[t][:-1]).to(hp_dtype) for t in BITS.keys()}


def dge_fwd(x, dtype):
    break_points = BREAK_POINTS[dtype].to(x.device)
    intervals = INTERVALS[dtype].to(x.device)
    # torch.searchsorted internally materializes non-contiguous inputs and emits
    # a performance warning.  Make that copy explicit so DGE callers can pass
    # axis-transposed QDQ views without noisy test output.
    if not x.is_contiguous():
        x = x.contiguous()
    idx = torch.searchsorted(break_points, x, right=False, out_int32=True) - 1
    idx = torch.where(idx < 0, 0, idx)
    idx = torch.where(idx >= len(break_points) - 1, len(break_points) - 2, idx)
    delta = intervals[idx]
    half_delta = delta / 2.
    mid_point = break_points[idx] + half_delta
    return half_delta**(1 - 1 / k) * torch.sign(x - mid_point) * (torch.abs(x - mid_point)**(1 / k)) + mid_point


def dge_bwd(x, dtype):
    break_points = BREAK_POINTS[dtype].to(x.device)
    intervals = INTERVALS[dtype].to(x.device)
    # See dge_fwd: searchsorted expects contiguous values for the fast path.
    if not x.is_contiguous():
        x = x.contiguous()
    idx = torch.searchsorted(break_points, x, right=False, out_int32=True) - 1
    idx = torch.where(idx < 0, 0, idx)
    idx = torch.where(idx >= len(break_points) - 1, len(break_points) - 2, idx)
    delta = intervals[idx]
    half_delta = delta / 2.
    mid_point = break_points[idx] + half_delta
    return torch.clamp(half_delta**(1 - 1 / k) * (torch.abs(x - mid_point)**(1 / k - 1)) / k, 0, 3.)


@triton.jit
def triton_pow(x, y):
    return tl.exp(tl.log(x) * y)


@triton.jit
def dge_bwd_triton(
    x,
    break_points,
    intervals,
    n_breaks: tl.constexpr = 255,
    k: tl.constexpr = 5,
):
    x = x.view(64 * 64)
    idx = tl.full([64 * 64], 0, dtype=tl.int32)
    for i in range(0, n_breaks - 1):
        idx_i = tl.full([1], i, dtype=tl.int32)
        bp = tl.gather(break_points, idx_i, axis=0)
        idx = tl.where(x >= bp, i, idx)

    delta = tl.gather(intervals, idx, axis=0)
    half_delta = delta * 0.5
    mid_point = tl.gather(break_points, idx, axis=0) + half_delta

    abs_diff = tl.abs(x - mid_point)
    pow1 = 1.0 / k
    pow2 = 1.0 - pow1
    safe_abs_diff = tl.where(abs_diff > 1e-7, abs_diff, 1e-7)
    out = triton_pow(half_delta, pow2) * triton_pow(safe_abs_diff, (pow1 - 1.0)) / k
    out = tl.clamp(out, 0.0, 3.0)
    return out.view(64, 64)


@triton.jit
def dge_bwd_kernel(
    x_ptr,
    break_points_ptr,
    intervals_ptr,
    out_ptr,
    n_breaks: tl.constexpr = 255,
    k: tl.constexpr = 5,
):
    x = tl.load(x_ptr + tl.arange(0, 64 * 64))

    break_points = tl.load(
        break_points_ptr + tl.arange(0, 256),
        mask=tl.arange(0, 256) < 255,
        other=0.0,
    )
    intervals = tl.load(
        intervals_ptr + tl.arange(0, 256),
        mask=tl.arange(0, 256) < 254,
        other=0.0,
    )

    out = dge_bwd_triton(x, break_points, intervals, n_breaks, k)

    tl.store(out_ptr + tl.arange(0, 64 * 64), out.view(64 * 64))


def dge_bwd_triton_wrapper(x, dtype):
    break_points = BREAK_POINTS[dtype].to(x.device)
    intervals = INTERVALS[dtype].to(x.device)

    out = torch.empty_like(x, dtype=hp_dtype)
    grid = (1,)

    dge_bwd_kernel[grid](
        x,
        break_points,
        intervals,
        out,
    )

    return out
