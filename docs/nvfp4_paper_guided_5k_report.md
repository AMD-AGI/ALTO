# Paper-Guided NVFP4 5K Validation

| Recipe | last-200 avg | gap vs BF16 | gap vs MXFP4 | best loss | final loss |
|---|---:|---:|---:|---:|---:|
| `bf16` | 7.6280 | +0.0000 | -0.0164 | 7.6250 | 7.6250 |
| `mxfp4_light` | 7.6444 | +0.0164 | +0.0000 | 7.6250 | 7.6562 |
| `nvfp4_3qdq_rne_tail2` | 7.6673 | +0.0394 | +0.0230 | 7.6250 | 7.6875 |
| `nvfp4_3qdq_rne_tail1` | 7.7050 | +0.0770 | +0.0606 | 7.6562 | 7.6875 |
| `nvfp4_3qdq_rne_tail2_h16` | 7.7192 | +0.0912 | +0.0748 | 7.6562 | 7.6875 |
| `nvfp4_3qdq_rne_tail1_h16` | 7.8070 | +0.1791 | +0.1627 | 7.7500 | 7.7812 |
| `nvfp4_3qdq_rne` | 8.0120 | +0.3841 | +0.3677 | 7.8750 | 8.0000 |

## Top NVFP4 Directions

1. `nvfp4_3qdq_rne_tail2`: tail avg `7.6673`, gap vs MXFP4 `+0.0230`
2. `nvfp4_3qdq_rne_tail1`: tail avg `7.7050`, gap vs MXFP4 `+0.0606`

## Interpretation

- `tail1/tail2` test the paper idea of keeping a small number of sensitive tail layers in BF16.
- `h16` variants additionally test Wgrad-only Hadamard with size `16`, which is the most paper-like insertion point for the current QDQ path.
- The best recipes from this sweep are the best paper-guided directions to carry forward into longer runs.
