import argparse

import torch
import triton
import triton.language as tl

import modeloptimizer.kernels.mxfp8.mxfp8_linear  # noqa: F401


@triton.jit
def repro_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsn,
    stride_bsk,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FP8_FORMAT: tl.constexpr,
):
    tl.assume(BLOCK_SIZE_K % QUANT_BLOCK_SIZE == 0)

    n_rep_k: tl.constexpr = BLOCK_SIZE_K // QUANT_BLOCK_SIZE
    ks: tl.constexpr = K // QUANT_BLOCK_SIZE

    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    num_blocks_k = tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(num_blocks_k):
        offs_k = i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        offs_k_scale = i * n_rep_k + tl.arange(0, n_rep_k)
        mask_k_scale = offs_k_scale < ks
        a_s_ptrs = a_s_ptr + offs_m[:, None] * stride_asm + offs_k_scale[None, :] * stride_ask
        b_s_ptrs = b_s_ptr + offs_n[:, None] * stride_bsn + offs_k_scale[None, :] * stride_bsk
        a_s = tl.load(a_s_ptrs, mask=mask_m[:, None] & mask_k_scale[None, :], other=1)
        b_s = tl.load(b_s_ptrs, mask=mask_n[:, None] & mask_k_scale[None, :], other=1)

        if FP8_FORMAT == 0:
            acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)
        else:
            acc = tl.dot_scaled(a, a_s, "e5m2", b, b_s, "e5m2", acc=acc, out_dtype=tl.float32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def make_scale_disparity_case(shape, ratio, fmt):
    m, n, k = shape
    torch.manual_seed(0)
    a = torch.randn((m, k), device="cuda", dtype=torch.float32) * 0.1
    b = torch.randn((k, n), device="cuda", dtype=torch.float32) * 0.1
    a[:, :32] *= ratio
    b[:32, :] *= ratio

    a_lp, a_s = torch.ops.modeloptimizer.convert_to_mxfp8(
        a, mxfp_format=fmt, axis=-1, is_2d_block=False
    )
    b_lp, b_s = torch.ops.modeloptimizer.convert_to_mxfp8(
        b, mxfp_format=fmt, axis=-2, is_2d_block=False
    )
    a_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        a_lp, a_s, output_dtype=torch.float32, axis=-1, is_2d_block=False
    )
    b_dq = torch.ops.modeloptimizer.convert_from_mxfp8(
        b_lp, b_s, output_dtype=torch.float32, axis=-2, is_2d_block=False
    )
    ref = a_dq @ b_dq
    return a_lp, a_s, b_lp, b_s, ref


def run_kernel(a_lp, a_s, b_lp, b_s, block_m, block_n, block_k):
    m, k = a_lp.shape
    kb, n = b_lp.shape
    assert kb == k
    c = torch.empty((m, n), device=a_lp.device, dtype=torch.float32)
    fp8_format = 0 if a_lp.dtype == torch.float8_e4m3fn else 1
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    repro_kernel[grid](
        a_lp,
        b_lp,
        c,
        a_s,
        b_s,
        *a_lp.stride(),
        b_lp.stride()[1],
        b_lp.stride()[0],
        *c.stride(),
        *a_s.stride(),
        b_s.stride()[1],
        b_s.stride()[0],
        M=m,
        N=n,
        K=k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        QUANT_BLOCK_SIZE=32,
        FP8_FORMAT=fp8_format,
    )
    return c


def summarize(shape, fmt, ratio, block_m, block_n, block_k):
    a_lp, a_s, b_lp, b_s, ref = make_scale_disparity_case(shape, ratio, fmt)
    cur = run_kernel(a_lp, a_s, b_lp, b_s, block_m, block_n, block_k)
    diff = (cur - ref).abs()
    return {
        "shape": shape,
        "fmt": fmt,
        "ratio": ratio,
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "allclose": bool(torch.allclose(cur, ref)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=1024.0)
    args = parser.parse_args()

    print("=== Minimal reproduction ===")
    print(summarize((256, 384, 64), "e4m3", args.ratio, 64, 64, 32))
    print(summarize((256, 384, 128), "e4m3", args.ratio, 64, 64, 32))
    print(summarize((256, 384, 128), "e5m2", args.ratio, 64, 64, 32))

    print("=== Tile shape scan for one failing case ===")
    for block_m in (16, 32, 64, 128):
        for block_n in (16, 32, 64, 128):
            print(summarize((256, 384, 128), "e4m3", args.ratio, block_m, block_n, 32))


if __name__ == "__main__":
    main()
