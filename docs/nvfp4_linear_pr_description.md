# Support NVFP4 Linear for low-precision training

## Summary

Enables end-to-end NVFP4 (NVIDIA FP4, E2M1 + E4M3 block scales) low-precision training by adding a Linear operator with full forward/backward autograd, wiring it into the dispatch layer so dense / MoE-Linear models can opt in via `precision="nvfp4"` without code changes, and porting the wgrad-path Hadamard and DGE recipes from MXFP4. On `llama3-debugmodel` (tail2 recipe), NVFP4 converges to within **+0.0007 loss** of the BF16 baseline over 10K steps; with the recommended `2D-weight + Hadamard + DGE` recipe, the gap shrinks to **+0.0003** in just 5K steps.

The path is exercised exactly as `swap_params` would in production — wrapper tensor preserved as the autograd graph leaf — and every wrapped `Linear` parameter is verified to receive gradients and actually update each step (regression test included).

## What's in this PR

Four commits, each a self-contained layer of the stack:

1. **`kernels: support nvfp_linear`** — `NVFP4LinearFunction` autograd op emulating a future pure-NVFP4 GEMM via 6-QDQ BF16 fallback. Separate quantized views for the fprop and wgrad reduction axes; SR on backward; optional 2D block scaling on either operand; optional per-tensor scale.

2. **`kernels: dispatch: support nvfp4`** — `NVFP4TrainingWeightWrapperTensor` under `alto/kernels/dispatch/`. Routes `F.linear` / `mm` / `addmm` / `matmul` through the NVFP4 path, hard-fails on `torch._grouped_mm` (grouped GEMM lives in a separate branch) rather than silently falling back to BF16. The dispatch **passes the wrapper tensor itself into the autograd function**, mirroring the MXFP4 pattern — critical because `module.weight.data` is a detached view; pre-unwrapping to `B._data` makes the autograd leaf `requires_grad=False` and silently drops every wrapped weight's gradient. A protocol-aligned regression test pins `param.grad is not None` and `weight L2 drift > 0` after one AdamW step to prevent this footgun from coming back.

3. **`kernels: nvfp_linear: add Hadamard and DGE recipes`** — two training recipes ported from MXFP4:
   - **Hadamard** on the wgrad path: `_to_nvfp4_then_scaled_mm` lazily builds a `HadamardTransform`; forward/backward left-multiply `H` into the axis=0 operand before QDQ. `(Hx)ᵀ(Hg) = xᵀg` holds in high precision, so the only effect is decorrelated per-block outliers at FP4 rounding.
   - **DGE** (bin-aware weight-gradient modulation): `_qdq(return_raw=True)` threads the packed FP4 + scales out of the quantizer; backward dequantizes with unit scales to recover pure bin values and multiplies `grad_weights` by `dge_bwd(w_fp4_values, float4_e2m1fn_x2)`.

4. **`kernels: nvfp_linear: address review feedback`** — API and test hygiene:
   - `TrainingOpConfig` gains the NVFP4-only `use_per_tensor_scale` field (wired through `LowPrecisionTrainingModifier` and the dispatch layer; default `False`, so non-NVFP4 schemes are unaffected).
   - `_to_nvfp4_then_linear` → `_to_nvfp4_then_scaled_mm` (name parity with MXFP4's `_to_mxfp4_then_scaled_mm`; docstring clarifies bias is applied by the dispatch layer outside).
   - `alto/kernels/fp4/nvfp4/__init__.py` stops re-exporting underscore-prefixed private functions; `NVFP4LinearFunction` is the only public entry.
   - Backward's `grad_inputs.view(*original_shape[:-1], -1)` is rewritten with the explicit `w_dq.shape[-1]` for readability.
   - `calc_snr` / `calc_cossim` are consolidated into a new `alto/kernels/fp4/testing_utils.py`; the per-family `tests/utils.py` re-export, so no downstream test call-sites change.
   - Test tree aligned with MXFP4: `run_llama3_e2e.py` and `test_nvfp_linear_e2e.py` removed; `test_nvfp_linear.py` cut to two tests (QDQ round-trip + autograd parity, 80 cases, ~4 s total).

## Key results

### End-to-end convergence (llama3-debugmodel, `tail_bf16_blocks=2`, seed=0)

| Recipe                     | Steps | tail_avg(200) | Δ vs BF16   |
|---------------------------|-------|---------------|-------------|
| BF16 baseline             | 10K   | 7.6252        | 0           |
| NVFP4 (plain)             | 10K   | 7.6259        | **+0.0007** |
| NVFP4 (all-NVFP4, tail0)  | 10K   | 7.6762        | +0.0510     |

### Recipe study at 5K (base = `use_2dblock_w=True`)

| Recipe                    | Hadamard | DGE | tail_avg | Δ vs BF16   |
|--------------------------|:--------:|:---:|----------|-------------|
| 1D weight + 1D activation | –        | –   | 7.6425   | +0.0164     |
| 2D weight + 1D activation | –        | –   | 7.6311   | +0.0050     |
| + Hadamard                | ✓        | –   | 7.6288   | +0.0027     |
| + DGE                     | –        | ✓   | 7.6316   | +0.0055     |
| **+ Hadamard + DGE**      | ✓        | ✓   | **7.6264** | **+0.0003** |

Weight-side 2D scaling and the two new recipes are additive; `Hadamard + DGE` is super-additive (−0.0047 vs base, larger than the sum of the two alone). The `Hadamard + DGE` recipe at 5K already matches the plain NVFP4 result at 10K.

### Recommended `TrainingOpConfig`

```python
TrainingOpConfig(
    precision="nvfp4",
    use_2dblock_x=False,
    use_2dblock_w=True,
    use_sr_grad=True,
    use_hadamard=True,
    use_dge=True,
    use_per_tensor_scale=False,
)
```

## Testing

| Suite                                          | Result                                        |
|-----------------------------------------------|-----------------------------------------------|
| `alto/kernels/fp4/nvfp4/tests/`               | 238 passed, 0 xfailed, 4.15 s                 |
| `alto/kernels/fp4/mxfp4/tests/` (quantization / attention / grouped-GEMM) | 397 passed, 2 xfailed (pre-existing)      |
| 5K E2E on `llama3-debugmodel`, all recipes    | PASS (all recipes meet protocol Steps 1–5)    |
| 10K BF16-vs-NVFP4 convergence (tail2 / tail0) | PASS                                          |

(16 pre-existing `ConvertTritonAMDGPUToLLVM` failures in `test_mxfp_linear.py::test_mxfp4_linear_autograd_function` on `gfx950` are unrelated to this PR; they fail at Triton kernel compile time and do not go through any code touched here.)

Dispatch-guard coverage (38 cases in `test_nvfp_dispatch_guards.py`):

- `torch._grouped_mm` under `scheme=nvfp4` raises `NotImplementedError`.
- Every supported knob combination (32 cases across 2Dx / 2Dw / SR / Hadamard / DGE) returns a valid BF16 output.
- Under production `swap_params`, `param.grad` lands on the wrapper (not `param._data.grad`) and one AdamW step moves the weight — verified for plain NVFP4, Hadamard-only, DGE-only, and Hadamard + DGE.

## Scope / non-goals

- Linear only; `torch._grouped_mm` is intentionally rejected and tracked in `zhitao/support-nvfp4-group-gemm`.
- Current path is BF16-simulated (`QDQ → BF16 GEMM`). Native FP4 MMA for NVFP4 (E4M3 block scales) will be added once the hardware path lands; the op-level API (`_qdq(return_raw=True)` and the axis-specific saved-tensor layout) already matches what a native kernel would consume.
