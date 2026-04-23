"""Diagnose the near-zero / relative-error problem in P2P BF16 vs NVFP4."""
import torch
from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_linear

torch.manual_seed(1234)
device = torch.device("cuda")

def prepare_data(shape, dtype):
    torch.manual_seed(1234)
    x = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.bernoulli(torch.ones_like(x) * 0.005)
    x = x + 100 * torch.randn_like(x) * mask
    return x

B, M, N, K = 4, 1024, 1024, 2048
dtype = torch.bfloat16

x = prepare_data((B, M, K), dtype)
w = prepare_data((N, K), dtype)

y_bf16  = torch.nn.functional.linear(x, w)
y_nvfp4 = _to_nvfp4_then_linear(x, w, use_2dblock_x=False, use_2dblock_w=False,
                                 use_sr_grad=False, use_per_tensor_scale=False)

diff = (y_nvfp4.float() - y_bf16.float()).abs()
yf   = y_bf16.float()
amax = yf.abs().max().item()

print("=" * 65)
print(f"Shape  : ({B}, {M}, {N}, {K})  dtype={dtype}")
print(f"K      : {K}  (high accumulation => many cancellations)")
print(f"Elements: {yf.numel():,}")
print()

# --- 1. |y_bf16| distribution
print("--- Distribution of |y_bf16| (amax = {:.0f}) ---".format(amax))
abs_yf = yf.abs()
for t in [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0]:
    n = (abs_yf < t * amax).sum().item()
    print(f"  |y| < {t:.0e} x amax  ({t*amax:>10.2f}):  {n:>8,}  ({100.0*n/yf.numel():5.2f}%)")
print()

# --- 2. Naive element-wise relative error (no threshold)
raw_rel = diff / (abs_yf + 1e-30)
print("--- Naive element-wise relative error (no threshold) ---")
for p in [50, 75, 90, 95, 99, 99.9, 100]:
    v = torch.quantile(raw_rel.reshape(-1).float(), p / 100.0).item()
    print(f"  p{p:5.1f} : {v:>12.2f}x")
print(f"  mean  : {raw_rel.mean().item():>12.4f}x   <-- inflated by near-zero denom")
print()

# --- 3. Thresholded relative error
for t in [0.001, 0.01, 0.05, 0.1]:
    mask = abs_yf > t * amax
    n_m = mask.sum().item()
    if n_m == 0:
        continue
    rel = (diff[mask] / abs_yf[mask])
    print(f"Threshold |y| > {t:.1%} x amax  ({n_m:,} elements kept = {100.0*n_m/yf.numel():.1f}%)")
    print(f"  mean relative error : {rel.mean().item():.6f}  ({rel.mean().item()*100:.4f}%)")
    print(f"  max  relative error : {rel.max().item():.6f}  ({rel.max().item()*100:.4f}%)")
    print()

# --- 4. Ten smallest-|y_bf16| elements to make the problem concrete
print("--- 10 elements with smallest |y_bf16| ---")
flat_abs   = abs_yf.reshape(-1)
flat_diff  = diff.reshape(-1)
flat_yb    = yf.reshape(-1)
flat_yn    = y_nvfp4.float().reshape(-1)
idxs       = flat_abs.argsort()[:10]
print(f"  {'idx':>8}  {'y_bf16':>14}  {'y_nvfp4':>14}  {'|diff|':>10}  {'|diff/y_bf16|':>14}")
print("  " + "-" * 64)
for i in idxs.tolist():
    yb = flat_yb[i].item()
    yn = flat_yn[i].item()
    ad = abs(yn - yb)
    re = ad / (abs(yb) + 1e-30)
    print(f"  {i:>8d}  {yb:>14.4f}  {yn:>14.4f}  {ad:>10.4f}  {re:>14.2f}x")

print()
print("=" * 65)
print("Key observation:")
print("  Near-zero y_bf16 values arise from K=2048 partial-sum cancellations.")
print("  For those elements, |diff / y_bf16| >> 1 even though |diff| is tiny.")
print("  This makes naive mean-relative-error a misleading metric.")
print("  Normalized MAE (MAE / amax) or thresholded relative error is reliable.")
