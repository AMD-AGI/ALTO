"""Diagnose why Route B E4M3 scale rounding breaks training."""
import torch
import math
import sys
sys.stdout.reconfigure(line_buffering=True)

from alto.kernels.fp4.nvfp4.nvfp_quantization import (
    convert_to_nvfp4, convert_from_nvfp4,
)


def round_scales_to_e4m3(scales):
    s_int = scales.view(torch.int32)
    s_int = (s_int + (1 << 19)) & 0xFFF00000
    return s_int.view(torch.float32)


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    x = torch.randn(512, 256, device=device, dtype=torch.bfloat16) * 0.5

    # === Test 1: No per-tensor scale ===
    data_lp, scales, pts = convert_to_nvfp4(x, axis=-1, block_size=16)
    scales_rounded = round_scales_to_e4m3(scales)

    x_dq_correct = convert_from_nvfp4(
        data_lp, scales, output_dtype=x.dtype, axis=-1, block_size=16)
    x_dq_rounded = convert_from_nvfp4(
        data_lp, scales_rounded, output_dtype=x.dtype, axis=-1, block_size=16)

    scale_rel_err = ((scales_rounded - scales).abs() / scales.clamp(min=1e-10)).mean()
    print("=== Without per-tensor scale ===")
    print(f"  Scale range:       [{scales.min():.6f}, {scales.max():.6f}]")
    print(f"  Scale mean rel err: {scale_rel_err:.4%}")
    print(f"  Dequant MAE (correct): {(x_dq_correct.float() - x.float()).abs().mean():.6f}")
    print(f"  Dequant MAE (rounded): {(x_dq_rounded.float() - x.float()).abs().mean():.6f}")
    print(f"  Mismatch MAE:          {(x_dq_correct - x_dq_rounded).abs().mean():.6f}")
    cos1 = torch.nn.functional.cosine_similarity(
        x_dq_correct.float().flatten(), x.float().flatten(), dim=0)
    cos2 = torch.nn.functional.cosine_similarity(
        x_dq_rounded.float().flatten(), x.float().flatten(), dim=0)
    print(f"  Cosine sim (correct):  {cos1:.8f}")
    print(f"  Cosine sim (rounded):  {cos2:.8f}")

    # === Test 2: WITH per-tensor scale ===
    data_lp2, scales2, pts2 = convert_to_nvfp4(
        x, axis=-1, block_size=16, dynamic_per_tensor_scale=True)
    scales2_rounded = round_scales_to_e4m3(scales2)

    x_dq2_correct = convert_from_nvfp4(
        data_lp2, scales2, output_dtype=x.dtype, axis=-1, block_size=16,
        per_tensor_scale=pts2)
    x_dq2_rounded = convert_from_nvfp4(
        data_lp2, scales2_rounded, output_dtype=x.dtype, axis=-1, block_size=16,
        per_tensor_scale=pts2)

    scale2_rel_err = ((scales2_rounded - scales2).abs() / scales2.clamp(min=1e-10)).mean()
    print()
    print("=== WITH per-tensor scale ===")
    print(f"  Per-tensor scale:   {pts2.item():.8f}")
    print(f"  Scale range:        [{scales2.min():.6f}, {scales2.max():.6f}]")
    print(f"  Scale mean rel err: {scale2_rel_err:.4%}")
    print(f"  Dequant MAE (correct): {(x_dq2_correct.float() - x.float()).abs().mean():.6f}")
    print(f"  Dequant MAE (rounded): {(x_dq2_rounded.float() - x.float()).abs().mean():.6f}")
    print(f"  Mismatch MAE:          {(x_dq2_correct - x_dq2_rounded).abs().mean():.6f}")
    cos3 = torch.nn.functional.cosine_similarity(
        x_dq2_correct.float().flatten(), x.float().flatten(), dim=0)
    cos4 = torch.nn.functional.cosine_similarity(
        x_dq2_rounded.float().flatten(), x.float().flatten(), dim=0)
    print(f"  Cosine sim (correct):  {cos3:.8f}")
    print(f"  Cosine sim (rounded):  {cos4:.8f}")

    # === Test 3: Check gradient signal ===
    print()
    print("=== Gradient signal test ===")
    x_test = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)
    w_test = torch.randn(64, 128, device=device, dtype=torch.bfloat16, requires_grad=True)

    # Normal QDQ path
    def qdq_normal(t, axis):
        d, s, p = convert_to_nvfp4(t, axis=axis, block_size=16)
        return convert_from_nvfp4(d, s, output_dtype=t.dtype, axis=axis, block_size=16)

    # Route B QDQ path (E4M3 rounding + per-tensor scale)
    def qdq_route_b(t, axis):
        d, s, p = convert_to_nvfp4(t, axis=axis, block_size=16, dynamic_per_tensor_scale=True)
        s_r = round_scales_to_e4m3(s)
        return convert_from_nvfp4(d, s_r, output_dtype=t.dtype, axis=axis, block_size=16,
                                  per_tensor_scale=p)

    # Forward with normal QDQ
    x1 = x_test.detach().clone().requires_grad_(True)
    w1 = w_test.detach().clone().requires_grad_(True)
    x1_dq = qdq_normal(x1, axis=-1)
    w1_dq = qdq_normal(w1, axis=-1)
    y1 = x1_dq @ w1_dq.T
    y1.sum().backward()
    grad_x_normal = x1.grad.clone()
    grad_w_normal = w1.grad.clone()

    # Forward with Route B QDQ
    x2 = x_test.detach().clone().requires_grad_(True)
    w2 = w_test.detach().clone().requires_grad_(True)
    x2_dq = qdq_route_b(x2, axis=-1)
    w2_dq = qdq_route_b(w2, axis=-1)
    y2 = x2_dq @ w2_dq.T
    y2.sum().backward()
    grad_x_rb = x2.grad.clone()
    grad_w_rb = w2.grad.clone()

    print(f"  grad_x normal magnitude: {grad_x_normal.abs().mean():.6f}")
    print(f"  grad_x route_b magnitude: {grad_x_rb.abs().mean():.6f}")
    print(f"  grad_w normal magnitude: {grad_w_normal.abs().mean():.6f}")
    print(f"  grad_w route_b magnitude: {grad_w_rb.abs().mean():.6f}")

    # Check if gradients are NaN or zero
    print(f"  grad_x route_b has NaN: {grad_x_rb.isnan().any()}")
    print(f"  grad_x route_b all zero: {(grad_x_rb == 0).all()}")
    print(f"  grad_w route_b has NaN: {grad_w_rb.isnan().any()}")
    print(f"  grad_w route_b all zero: {(grad_w_rb == 0).all()}")

    # === Test 4: The compound effect — 6 QDQ operations ===
    print()
    print("=== Compound effect: 6x QDQ ===")
    t = torch.randn(512, 256, device=device, dtype=torch.bfloat16) * 0.5
    t_normal = t.clone()
    t_rb = t.clone()
    for i in range(6):
        d, s, p = convert_to_nvfp4(t_normal, axis=-1, block_size=16)
        t_normal = convert_from_nvfp4(d, s, output_dtype=t.dtype, axis=-1, block_size=16)

        d, s, p = convert_to_nvfp4(t_rb, axis=-1, block_size=16, dynamic_per_tensor_scale=True)
        s_r = round_scales_to_e4m3(s)
        t_rb = convert_from_nvfp4(d, s_r, output_dtype=t.dtype, axis=-1, block_size=16,
                                  per_tensor_scale=p)

    cos_n = torch.nn.functional.cosine_similarity(
        t_normal.float().flatten(), t.float().flatten(), dim=0)
    cos_r = torch.nn.functional.cosine_similarity(
        t_rb.float().flatten(), t.float().flatten(), dim=0)
    mae_n = (t_normal.float() - t.float()).abs().mean()
    mae_r = (t_rb.float() - t.float()).abs().mean()
    print(f"  After 6x QDQ (normal):  cosine={cos_n:.6f}  MAE={mae_n:.6f}")
    print(f"  After 6x QDQ (route_b): cosine={cos_r:.6f}  MAE={mae_r:.6f}")
    print(f"  Signal ratio (normal/route_b MAE): {mae_r/mae_n:.2f}x worse")


if __name__ == "__main__":
    main()
