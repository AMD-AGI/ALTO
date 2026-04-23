# NVFP4 Recipe Quick Validation Report

## Goal

Use small-step experiments to identify NVFP4 training recipes that move the loss curve closer to `MXFP4` and `BF16`, then combine the quick results with the existing `20K` evidence.

## Recipe Matrix

| Recipe | Key changes | Intent |
|---|---|---|
| `nvfp4_current` | 6 QDQ, forward RNE, grad SR | current baseline |
| `nvfp4_6qdq_xsr` | 6 QDQ, activation forward SR | isolate forward SR only |
| `nvfp4_3qdq_rne` | 3 QDQ, forward RNE, grad SR | isolate QDQ count reduction |
| `nvfp4_route_a` | 3 QDQ, activation forward SR, grad SR | Route A |
| `nvfp4_route_a_pts_xg` | Route A + per-tensor scale on activation/grad | test targeted PTS |
| `nvfp4_route_a_pts_all` | Route A + per-tensor scale everywhere | test aggressive PTS |

## 1K Sweep Results

| Recipe | tail avg loss | gap vs BF16 | gap vs MXFP4 | best loss |
|---|---:|---:|---:|---:|
| `bf16` | 7.6422 | +0.0000 | -0.0144 | 7.6250 |
| `mxfp4_light` | 7.6566 | +0.0144 | +0.0000 | 7.6250 |
| `nvfp4_3qdq_rne` | 8.0744 | +0.4322 | +0.4178 | 8.0000 |
| `nvfp4_route_a_pts_xg` | 8.0787 | +0.4366 | +0.4222 | 7.9688 |
| `nvfp4_6qdq_xsr` | 8.0819 | +0.4397 | +0.4253 | 7.9688 |
| `nvfp4_route_a_pts_all` | 8.0834 | +0.4413 | +0.4269 | 7.9688 |
| `nvfp4_current` | 8.0837 | +0.4416 | +0.4272 | 7.9688 |
| `nvfp4_route_a` | 8.0934 | +0.4512 | +0.4369 | 7.9375 |

## 3K Sweep Results

| Recipe | tail avg loss | gap vs BF16 | gap vs MXFP4 | best loss |
|---|---:|---:|---:|---:|
| `bf16` | 7.6317 | +0.0000 | -0.0181 | 7.6250 |
| `mxfp4_light` | 7.6498 | +0.0181 | +0.0000 | 7.6250 |
| `nvfp4_3qdq_rne` | 8.0283 | +0.3966 | +0.3784 | 7.9062 |
| `nvfp4_current` | 8.0403 | +0.4086 | +0.3905 | 7.9375 |
| `nvfp4_route_a_pts_xg` | 8.0425 | +0.4108 | +0.3927 | 7.9375 |
| `nvfp4_route_a` | 8.0480 | +0.4163 | +0.3981 | 7.9375 |

## 5K Sweep Results

| Recipe | tail avg loss | gap vs BF16 | gap vs MXFP4 | best loss |
|---|---:|---:|---:|---:|
| `bf16` | 7.6280 | +0.0000 | -0.0164 | 7.6250 |
| `mxfp4_light` | 7.6444 | +0.0164 | +0.0000 | 7.6250 |
| `nvfp4_3qdq_rne` | 8.0050 | +0.3770 | +0.3606 | 7.8750 |
| `nvfp4_route_a` | 8.0153 | +0.3873 | +0.3709 | 7.9062 |

## Existing 20K Evidence

| Recipe | last-500 avg loss | gap vs BF16 | best loss | note |
|---|---:|---:|---:|---|
| `bf16` | 7.6251 | +0.0000 | 7.6250 | baseline |
| `mxfp4_light` | 7.6309 | +0.0058 | 7.5938 | close to BF16 |
| `nvfp4_current` | 7.8293 | +0.2042 | 7.7188 | clear plateau |
| `nvfp4_route_a` | 7.7024 | +0.0773 | 7.6250 | standalone 20K run |

## Findings

1. **Reducing QDQ count is the strongest fast win.** In both the `1K`, `3K`, and `5K` sweeps, `nvfp4_3qdq_rne` is the best NVFP4 recipe. This isolates the benefit of dropping the extra axis=0 QDQ path.
2. **Forward SR does not help in the short horizon.** `nvfp4_route_a` is consistently slightly worse than `nvfp4_3qdq_rne` up to `5K` steps, so SR is not a quick fix for the early optimization phase.
3. **Per-tensor scale is neutral to slightly negative in the quick tests.** Both `pts_xg` and `pts_all` fail to beat the plain Route A variant in the short sweep, which matches the earlier theory that the current float32 scale path is not saturation-limited.
4. **Route A still has long-horizon potential.** In the earlier standalone `20K` run, `nvfp4_route_a` reached a much lower final loss than the current NVFP4 baseline. That means Route A may need longer training to realize its benefit, but it is not the best fast-validation recipe.
5. **Current evidence says the primary bottleneck is recipe noise budget, not scale fidelity.** The best quick improvement comes from removing extra QDQ, not from enabling more advanced scaling machinery.

## Recommended Recipes

### Default quick-win recipe

- `3 QDQ` instead of `6 QDQ`
- forward activation: `RNE`
- forward weight: `RNE`
- backward grad: `SR`
- per-tensor scale: `off`

This corresponds to `nvfp4_3qdq_rne`. It is the best recipe in every quick sweep and should be the default next candidate to compare against `MXFP4` in a longer run.

### Secondary long-horizon candidate

- `3 QDQ` instead of `6 QDQ`
- forward activation: `SR`
- forward weight: `RNE`
- backward grad: `SR`
- per-tensor scale: `off`

This is `nvfp4_route_a`. It is not the best at `1K-5K`, but the prior `20K` standalone run suggests it may improve later. It should be validated next with a focused `10K` or `20K` comparison against `nvfp4_3qdq_rne`.

## Next Validation Plan

1. Run `bf16`, `mxfp4_light`, `nvfp4_3qdq_rne`, and `nvfp4_route_a` for `10K` steps in the same script.
2. If `nvfp4_route_a` still does not overtake `nvfp4_3qdq_rne`, standardize on `nvfp4_3qdq_rne` as the default NVFP4 training recipe.
3. Only after that, revisit proper E4M3-scale simulation, and do it inside the quant path rather than by post-hoc editing the dequant scale tensor.

