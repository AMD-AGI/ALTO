import torch
import triton
import triton.language as tl

import alto.kernels.mxfp8.mxfp8_quantization  # noqa: F401


@triton.jit
def dot_scaled_kernel(a_ptr, b_ptr, a_s_ptr, b_s_ptr, c_ptr, K: tl.constexpr):
    offs_m = tl.arange(0, 64)
    offs_n = tl.arange(0, 64)

    acc = tl.zeros((64, 64), dtype=tl.float32)
    for k0 in range(0, K, 32):
        offs_k = k0 + tl.arange(0, 32)
        scale_k = k0 // 32

        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * 64 + offs_n[None, :])
        a_s = tl.load(a_s_ptr + offs_m[:, None] * (K // 32) + scale_k)
        b_s = tl.load(b_s_ptr + scale_k * 64 + offs_n[:, None])

        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc=acc, out_dtype=tl.float32)

    tl.store(c_ptr + offs_m[:, None] * 64 + offs_n[None, :], acc)


def run_case(name, k):
    m, n = 64, 64
    torch.manual_seed(0)

    a = torch.randn((m, k), device="cuda") * 0.1
    b = torch.randn((k, n), device="cuda") * 0.1
    a[:, :32] *= 1024
    b[:32, :] *= 1024

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
    dot_scaled_kernel[(1,)](a_lp, b_lp, a_s, b_s, out, K=k)

    ref = a_dq @ b_dq
    diff = (out - ref).abs()
    print(f"== {name} ==")
    print(f"allclose={torch.allclose(out, ref)}")
    print(f"max_diff={diff.max().item():.6f}")
    print(f"mean_diff={diff.mean().item():.6f}")


def main():
    run_case("single tl.dot_scaled, K=32", 32)
    run_case("K-chain tl.dot_scaled, K=128", 128)


if __name__ == "__main__":
    main()
