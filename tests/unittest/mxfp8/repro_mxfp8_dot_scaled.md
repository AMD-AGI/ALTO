## MXFP8 `tl.dot_scaled` Repro

Minimal reproducer for Triton's MXFP8 `tl.dot_scaled`.

The script builds `e4m3` MXFP8 GEMM inputs, runs one Triton kernel that calls
`tl.dot_scaled`, then compares the result with:

```python
convert_from_mxfp8(a) @ convert_from_mxfp8(b)
```

### Cases

The same `dot_scaled_kernel` covers both cases:

1. `K=32`: one `tl.dot_scaled` call.
2. `K=128`: four `tl.dot_scaled` calls accumulated with `BLOCK_K=32`.

### How to run

```bash
python3 repro_mxfp8_dot_scaled.py
```

Run it in the same CDNA4 environment used for the MXFP8 kernel tests.

### Expected signal

The script prints `allclose`, `max_diff`, and `mean_diff` for both cases.
