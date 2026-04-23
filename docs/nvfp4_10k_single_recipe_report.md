# NVFP4 10K Single-Recipe Validation

| Recipe | last-200 avg | gap vs BF16 | gap vs MXFP4 | best loss | final loss |
|---|---:|---:|---:|---:|---:|
| `bf16` | 7.6253 | +0.0000 | -0.0122 | 7.6250 | 7.6250 |
| `mxfp4_light` | 7.6375 | +0.0122 | +0.0000 | 7.6250 | 7.6250 |
| `nvfp4_route_a` | 7.9352 | +0.3098 | +0.2977 | 7.8125 | 7.9688 |
| `nvfp4_3qdq_rne` | 7.9744 | +0.3491 | +0.3369 | 7.8438 | 8.0000 |

## Conclusions

1. `nvfp4_route_a` overtakes `nvfp4_3qdq_rne` by `10K` steps, confirming it is the better long-horizon recipe.
2. `nvfp4_3qdq_rne` is still the best short-horizon recipe up to `5K` and is useful for fast iteration.
3. The practical default should depend on the target horizon:
   - quick screening / debugging: `nvfp4_3qdq_rne`
   - longer training aiming to approach MXFP4/BF16: `nvfp4_route_a`

