## MXFP8 `tl.dot_scaled` Repro

Minimal reproducer for the MXFP8 `tl.dot_scaled` multi-block accuracy issue.

The script builds `e4m3` MXFP8 GEMM inputs, runs one Triton kernel that calls
`tl.dot_scaled`, then compares the result with:

```python
convert_from_mxfp8(a) @ convert_from_mxfp8(b)
```

### Cases

The same `dot_scaled_kernel` covers three cases:

1. `K=32, DOT_K=32`: baseline, one `tl.dot_scaled` call over one quant block.
2. `K=128, DOT_K=32`: safe path, four `tl.dot_scaled` calls, each over one quant block.
3. `K=128, DOT_K=128`: problem path, one `tl.dot_scaled` call spanning four
   32-wide quant blocks.

The inputs include outlier-heavy 32-wide K blocks to make scale disparity visible.

### How to run

```bash
python3 repro_mxfp8_dot_scaled.py
```

Run it in the same CDNA4 environment used for the MXFP8 kernel tests.

### Expected signal

The problem path should show a noticeably larger error than the safe path. The script
prints `allclose`, `max_diff`, `mean_diff`, and `relative_max_diff` for each case.
