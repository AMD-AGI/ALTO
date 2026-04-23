"""
Diagnose WHY so many near-zero values appear in GEMM output.
Three root causes are analyzed:
  A. Statistical: random zero-mean inputs -> expected output ~0
  B. K-accumulation: more terms -> more cancellation -> closer to true 0
  C. BF16 precision: the dynamic range collapses small values to exact 0
"""
import torch

torch.manual_seed(1234)
device = torch.device("cuda")


def stats(y, label):
    yf = y.float()
    amax = yf.abs().max().item()
    near_zero = (yf.abs() < 1e-2 * amax).float().mean().item()
    exact_zero = (yf == 0).float().mean().item()
    print(f"  {label}")
    print(f"    amax          = {amax:>12.2f}")
    print(f"    near-zero  (<1% amax)  = {near_zero*100:5.1f}%  of elements")
    print(f"    exact-zero (== 0)      = {exact_zero*100:5.1f}%  of elements")
    print()


print("=" * 65)
print("CAUSE A: zero-mean inputs cause expected GEMM output to be ~0")
print("=" * 65)
print()
print("For i.i.d. zero-mean x[m,k] and w[n,k]:")
print("  y[m,n] = sum_{k} x[m,k]*w[n,k]")
print("  E[y[m,n]] = sum_{k} E[x]*E[w] = 0   (by independence, zero-mean)")
print()

# Show that output mean is near 0
K = 256
x = torch.randn(1, 64, K, dtype=torch.float32, device=device)
w = torch.randn(64, K, dtype=torch.float32, device=device)
y = torch.nn.functional.linear(x, w)
print(f"Example (F32, K={K}): output mean = {y.mean().item():.6f}   "
      f"(near 0 as expected for zero-mean inputs)")
print(f"Output std  = {y.std().item():.4f}  (non-trivial spread)")
print()

print("  => Most outputs are DISTRIBUTED AROUND 0,")
print("     so a large fraction falls close to zero just by probability.")
print()

print("=" * 65)
print("CAUSE B: larger K amplifies cancellation -> smaller output values")
print("=" * 65)
print()
print("By CLT: y[m,n] ~ N(0, K * Var(x)*Var(w))")
print("  -> output std grows as sqrt(K)")
print("  -> amax grows as sqrt(K)")
print("  -> but the RATIO of near-zero to amax stays roughly constant")
print("  -> however BF16 precision (7 mantissa bits) limits small-value resolution")
print()

print("K-sweep (BF16, same relative threshold = 1% of amax):")
print(f"  {'K':>6}  {'amax':>12}  {'std':>10}  {'near-zero%':>12}")
for K in [64, 128, 256, 512, 1024, 2048]:
    x = torch.randn(1, 256, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(256, K, dtype=torch.bfloat16, device=device)
    y = torch.nn.functional.linear(x, w).float()
    amax = y.abs().max().item()
    std  = y.std().item()
    nz   = (y.abs() < 0.01 * amax).float().mean().item() * 100
    print(f"  {K:>6}  {amax:>12.1f}  {std:>10.1f}  {nz:>11.1f}%")
print()

print("=" * 65)
print("CAUSE C: BF16 quantization maps small values to exact 0")
print("=" * 65)
print()
print("BF16 has only 7 mantissa bits -> effective resolution ~ 1/128 of amax")
print("Values within [-amax/128, +amax/128] are often rounded to 0 in BF16")
print()

# Compare F32 vs BF16 for same data
K = 2048
x_f32 = torch.randn(4, 1024, K, dtype=torch.float32, device=device)
w_f32 = torch.randn(1024, K, dtype=torch.float32, device=device)
x_bf16 = x_f32.bfloat16()
w_bf16 = w_f32.bfloat16()

y_f32  = torch.nn.functional.linear(x_f32, w_f32)
y_bf16 = torch.nn.functional.linear(x_bf16, w_bf16).float()

amax_f32  = y_f32.abs().max().item()
amax_bf16 = y_bf16.abs().max().item()
exact_z_f32  = (y_f32  == 0).float().mean().item() * 100
exact_z_bf16 = (y_bf16 == 0).float().mean().item() * 100
nz_f32  = (y_f32.abs()  < 0.01 * amax_f32).float().mean().item()  * 100
nz_bf16 = (y_bf16.abs() < 0.01 * amax_bf16).float().mean().item() * 100

print(f"Large GEMM (4×1024×1024, K={K}):")
print(f"  {'':10}  {'F32 accum':>12}  {'BF16':>12}")
print(f"  {'amax':10}  {amax_f32:>12.1f}  {amax_bf16:>12.1f}")
print(f"  {'exact zero':10}  {exact_z_f32:>11.2f}%  {exact_z_bf16:>11.2f}%")
print(f"  {'< 1% amax':10}  {nz_f32:>11.2f}%  {nz_bf16:>11.2f}%")
print()
print("  BF16's coarse grid rounds many small-but-nonzero values to exact 0.")
print()

print("=" * 65)
print("CAUSE D: test data design reinforces the problem")
print("=" * 65)
print()
print("The test uses: torch.randn + 0.5% sparse 100x outliers")
print("The outliers make amax MUCH larger, while the bulk stays near 0.")
print()

x_plain = torch.randn(4, 1024, 2048, dtype=torch.bfloat16, device=device)
w_plain = torch.randn(1024, 2048, dtype=torch.bfloat16, device=device)

mask = torch.bernoulli(torch.ones_like(x_plain) * 0.005)
x_outlier = x_plain + 100 * torch.randn_like(x_plain) * mask

y_plain   = torch.nn.functional.linear(x_plain, w_plain).float()
y_outlier = torch.nn.functional.linear(x_outlier, w_plain).float()

amax_plain   = y_plain.abs().max().item()
amax_outlier = y_outlier.abs().max().item()
nz_plain     = (y_plain.abs()   < 0.01 * amax_plain).float().mean().item()   * 100
nz_outlier   = (y_outlier.abs() < 0.01 * amax_outlier).float().mean().item() * 100

print(f"  {'':20}  {'plain randn':>12}  {'randn + outliers':>16}")
print(f"  {'amax':20}  {amax_plain:>12.1f}  {amax_outlier:>16.1f}")
print(f"  {'< 1% amax':20}  {nz_plain:>11.1f}%  {nz_outlier:>15.1f}%")
print()
print("  Outliers inflate amax ~10-100x, making the '< 1% amax' bar MUCH higher.")
print("  The bulk of the output (generated by non-outlier inputs) falls below it.")
print()

print("=" * 65)
print("SUMMARY")
print("=" * 65)
print("""
The high fraction of near-zero outputs is caused by ALL FOUR factors together:

  A. STATISTICAL:  random zero-mean inputs -> E[output] = 0
     -> output distribution is centered on 0 by construction

  B. ACCUMULATION: K=2048 partial sums -> high cancellation rate
     -> central-limit behavior pushes most values toward 0

  C. BF16 PRECISION: 7-bit mantissa (~1/128 relative precision)
     -> values < amax/128 are below BF16's resolution floor,
        many are rounded to exact 0

  D. TEST DATA: 0.5% sparse 100x outliers in input x
     -> output amax is dominated by outlier-amplified rows
     -> the relative threshold 'amax * 1%' is much higher than
        the typical non-outlier output magnitude

These four factors compound: zero-mean data + large K cancellation +
BF16 resolution limits + outlier-inflated amax = 95%+ of outputs near 0.

This is NOT a bug in the op. It IS a design issue for relative-error testing:
naive |diff/y_bf16| is ill-defined when y_bf16 ~= 0 through accumulation.
""")
