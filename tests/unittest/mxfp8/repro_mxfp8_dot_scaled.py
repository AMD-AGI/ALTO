import torch
import triton
import triton.language as tl

import alto.kernels.mxfp8.mxfp8_quantization  # noqa: F401


@triton.jit
def dot_scaled_kernel(a_ptr, b_ptr, a_s_ptr, b_s_ptr, c_ptr, K: tl.constexpr, DOT_K: tl.constexpr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)
    scale_count: tl.constexpr = K // 32
    dot_scale_count: tl.constexpr = DOT_K // 32

    acc = tl.zeros((64, 64), dtype=tl.float32)
    for k0 in range(0, K, DOT_K):
        offs_k = k0 + tl.arange(0, DOT_K)
        offs_scale = k0 // 32 + tl.arange(0, dot_scale_count)

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * 64 + offs_n[None, :])
        a_s = tl.load(a_s_ptr + offs_m[:, None] * scale_count + offs_scale[None, :])
        b_s = tl.load(b_s_ptr + offs_n[:, None] + offs_scale[None, :] * 64)

        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)

    tl.store(c_ptr + offs_m[:, None] * 64 + offs_n[None, :], acc)


def run_case(name, k, dot_k):
    m, n = 64, 64
    torch.manual_seed(0)

    a = torch.randn((m, k), device="cuda") * 0.1
    b = torch.randn((k, n), device="cuda") * 0.1
    # Make each 32-wide quant block use a different scale range.
    # This stresses the DOT_K=128 path, where one tl.dot_scaled spans multiple blocks.
    quant_block_ratios = (1024, 1, 256, 16)
    for block_idx, ratio in enumerate(quant_block_ratios):
        start = block_idx * 32
        end = min(start + 32, k)
        if start < k:
            a[:, start:end] *= ratio
            b[start:end, :] *= ratio

    a_lp, a_s = torch.ops.alto.convert_to_mxfp8(
        a, mxfp_format="e4m3", axis=-1, is_2d_block=False
    )
    b_lp, b_s = torch.ops.alto.convert_to_mxfp8(
        b, mxfp_format="e4m3", axis=-2, is_2d_block=False
    )
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
    print(f"== {name} ==")
    print(f"allclose={torch.allclose(out, ref)}")
    print(f"max_diff={diff.max().item():.6f}")
    print(f"mean_diff={diff.mean().item():.6f}")
    print(f"relative_max_diff={(diff.max() / ref_max).item():.6f}")


def main():
    run_case("baseline: K=32, DOT_K=32", 32, 32)
    run_case("safe path: K=128, DOT_K=32", 128, 32)
    run_case("problem path: K=128, DOT_K=128", 128, 128)


if __name__ == "__main__":
    main()
