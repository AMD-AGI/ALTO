# NVFP4 Phase1 Tail Sweep 5K

| Recipe | last-200 avg | gap vs BF16 | gap vs MXFP4 | best loss | final loss |
|---|---:|---:|---:|---:|---:|
| `bf16` | 7.6280 | +0.0000 | -0.0164 | 7.6250 | 7.6250 |
| `mxfp4_light` | 7.6444 | +0.0164 | +0.0000 | 7.6250 | 7.6562 |
| `nvfp4_3qdq_rne_tail2` | 7.6673 | +0.0394 | +0.0230 | 7.6250 | 7.6875 |
| `nvfp4_3qdq_rne_tail3` | 7.6731 | +0.0452 | +0.0287 | 7.6250 | 7.6562 |
| `nvfp4_route_a_tail4` | 7.6783 | +0.0503 | +0.0339 | 7.6250 | 7.6875 |
| `nvfp4_3qdq_rne_tail4` | 7.6797 | +0.0517 | +0.0353 | 7.6562 | 7.6875 |
| `nvfp4_route_a_tail3` | 7.6809 | +0.0530 | +0.0366 | 7.6250 | 7.6875 |
| `nvfp4_route_a_tail2` | 7.6825 | +0.0545 | +0.0381 | 7.6250 | 7.6875 |

## Winners

- 3QDQ line winner: `nvfp4_3qdq_rne_tail2` with tail avg `7.6673`
- RouteA line winner: `nvfp4_route_a_tail4` with tail avg `7.6783`
