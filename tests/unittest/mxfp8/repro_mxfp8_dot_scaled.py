import torch
import triton
import triton.language as tl

import alto.kernels.mxfp8.mxfp8_quantization  # noqa: F401


BLOCK_M = 64
BLOCK_N = 64
QUANT_BLOCK_SIZE = 32
OUTLIER_BLOCK_RATIOS = (1024, 1, 256, 16)


@triton.jit
def dot_scaled_kernel(a_ptr, b_ptr, a_s_ptr, b_s_ptr, c_ptr, K: tl.constexpr, DOT_K: tl.constexpr):
    block_m: tl.constexpr = 64
    block_n: tl.constexpr = 64
    quant_block_size: tl.constexpr = 32
    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    scale_count: tl.constexpr = K // quant_block_size
    dot_scale_count: tl.constexpr = DOT_K // quant_block_size

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for block_start in range(0, K, DOT_K):
        offs_k = block_start + tl.arange(0, DOT_K)
        offs_scale = block_start // quant_block_size + tl.arange(0, dot_scale_count)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * block_n + offs_n[None, :])
        a_s = tl.load(a_s_ptr + offs_m[:, None] * scale_count + offs_scale[None, :])
        b_s = tl.load(b_s_ptr + offs_n[:, None] + offs_scale[None, :] * block_n)

        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)

    tl.store(c_ptr + offs_m[:, None] * block_n + offs_n[None, :], acc)


def run_case(name, k, dot_k):
    m, n = BLOCK_M, BLOCK_N
    torch.manual_seed(0)

    a = torch.randn((m, k), device="cuda") * 0.1
    b = torch.randn((k, n), device="cuda") * 0.1
    # Make each 32-wide quant block use a different scale range.
    # This stresses the DOT_K=128 path, where one tl.dot_scaled spans multiple blocks.
    for block_idx, ratio in enumerate(OUTLIER_BLOCK_RATIOS):
        start = block_idx * QUANT_BLOCK_SIZE
        end = start + QUANT_BLOCK_SIZE
        if start < k:
            a[:, start:end] *= ratio
            b[start:end, :] *= ratio

    a_lp, a_s = torch.ops.alto.convert_to_mxfp8(
        a, mxfp_format="e4m3", axis=-1, is_2d_block=False
    )
    b_lp, b_s = torch.ops.alto.convert_to_mxfp8(
        b, mxfp_format="e4m3", axis=-2, is_2d_block=False
    )
    # axis=-2 conversion returns transposed-stride B tensors. This repro uses
    # a minimal pointer-arithmetic kernel, so normalize B to isolate dot_scaled.
    b_lp = b_lp.contiguous()
    b_s = b_s.contiguous()
    a_dq = torch.ops.alto.convert_from_mxfp8(
        a_lp, a_s, output_dtype=torch.float32, axis=-1, is_2d_block=False
    )
    b_dq = torch.ops.alto.convert_from_mxfp8(
        b_lp, b_s, output_dtype=torch.float32, axis=-2, is_2d_block=False
    )

    out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    dot_scaled_kernel[(1,)](a_lp, b_lp, a_s, b_s, out, K=k, DOT_K=dot_k)

    ref = a_dq @ b_dq
    diff = (out - ref).abs()
    ref_max = ref.abs().max()
    max_diff = diff.max()
    mean_diff = diff.mean()
    relative_max_diff = max_diff / ref_max

    print(f"== {name} ==")
    print(f"allclose={torch.allclose(out, ref, atol=1e-4, rtol=1e-4)}")
    print(f"max_diff={max_diff.item():.6f}")
    print(f"mean_diff={mean_diff.item():.6f}")
    print(f"relative_max_diff={relative_max_diff.item():.6f}")
    return max_diff.item(), mean_diff.item(), relative_max_diff.item()


def main():
    cases = (
        ("baseline", 32, 32),
        ("safe path", 128, 32),
        ("problem path: spans 2 quant blocks", 128, 64),
        ("problem path: spans 4 quant blocks", 128, 128),
    )
    mean_diffs = {}
    for name, k, dot_k in cases:
        _, mean_diff, _ = run_case(f"{name}: K={k}, DOT_K={dot_k}", k, dot_k)
        mean_diffs[dot_k] = mean_diff

    safe_mean_diff = mean_diffs[QUANT_BLOCK_SIZE]
    print(f"dot_k_64_vs_safe_mean_diff_ratio={mean_diffs[64] / safe_mean_diff:.2f}x")
    print(f"dot_k_128_vs_safe_mean_diff_ratio={mean_diffs[128] / safe_mean_diff:.2f}x")


if __name__ == "__main__":
    main()
