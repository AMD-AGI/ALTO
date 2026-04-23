# NVFP4 Tail vs Weight-2D 20K Comparison

| Recipe | last-500 avg | tail std | best loss | final loss |
|---|---:|---:|---:|---:|
| `nvfp4_tail2` | 7.6501 | 0.0135 | 7.6250 | 7.6562 |
| `nvfp4_tail3` | 7.6567 | 0.0130 | 7.5938 | 7.6562 |
| `nvfp4_tail2_w2d` | 7.6726 | 0.0170 | 7.6250 | 7.6562 |
| `nvfp4_tail3_w2d` | 7.6742 | 0.0172 | 7.6250 | 7.6875 |

## Conclusion

- Compare `tail2` vs `tail2_w2d` directly and `tail3` vs `tail3_w2d` directly to judge whether 2D weight scaling is beneficial on the current AMD + QDQ path.
