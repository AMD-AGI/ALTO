# NVFP4 Final 20K Report

## Final 8-Way Table

| Recipe | last-500 avg | tail std | best loss | final loss |
|---|---:|---:|---:|---:|
| `BF16` | 7.6251 | 0.0020 | 7.6250 | 7.6250 |
| `MXFP4 BF16` | 7.6307 | 0.0121 | 7.5938 | 7.6250 |
| `MXFP4 native` | 7.6322 | 0.0132 | 7.5938 | 7.6562 |
| `NVFP4 3QDQ+tail2` | 7.6529 | 0.0134 | 7.6250 | 7.6562 |
| `NVFP4 3QDQ+tail3` | 7.6562 | 0.0129 | 7.5938 | 7.6562 |
| `NVFP4 3QDQ+tail4` | 7.6660 | 0.0161 | 7.6250 | 7.6875 |
| `NVFP4 RouteA+tail3` | 7.6696 | 0.0163 | 7.6250 | 7.6875 |
| `NVFP4 RouteA+tail4` | 7.6754 | 0.0188 | 7.6250 | 7.6875 |

## Selection Logic

- Top 5 NVFP4 recipes were selected from Phase 1 and Phase 2 based on 5K validation loss.
- `MXFP4 native` is the current AMD-native low-precision baseline.
- `MXFP4 BF16` is the same MXFP4 quantization recipe but forced through the QDQ + BF16 GEMM path.

Plot: `/home/zhitwang/agi-model-opt/nvfp4_final_20k_8way.png`
