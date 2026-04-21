## MXFP8 `tl.dot_scaled` Repro

This reproducer shows a specific MXFP8 accuracy issue on CDNA4 when using Triton's
`tl.dot_scaled` in a fused GEMM path.

### What it shows

1. `BLOCK_SIZE_K=32` is necessary.
   If one `tl.dot_scaled` call spans multiple 32-wide quant blocks, the error grows
   sharply on outlier-heavy inputs.
2. Even with `BLOCK_SIZE_K=32`, long K-chain accumulation can still diverge from the
   `dequantize -> matmul` reference for MXFP8.
3. Changing `BLOCK_SIZE_M/N` does not materially change the error, so this does not
   look like a tile-shape tuning problem.

### How to run

```bash
python3 repro_mxfp8_dot_scaled.py
```

Run it inside the same CDNA4 Docker environment used for the MXFP8 kernel tests.

### Expected signal

- `(256, 384, 64)` with large scale disparity should still pass `torch.allclose`
- `(256, 384, 128)` with the same setup should fail `torch.allclose`
- scanning `BLOCK_SIZE_M/N` across `16/32/64/128` should produce essentially the same
  `max_diff` / `mean_diff`

### Related observation

Triton exposes `lhs_k_pack` / `rhs_k_pack` in `tl.dot_scaled`, but FP8 cannot use the
same non-K packing mode that MXFP4 uses. Probing `e4m3` / `e5m2` with either flag set
to `False` fails compilation with:

```text
only mxfp4 inputs can be packed along a dimension different than K
```

So MXFP8 does not have a corresponding "mxfp4-style packed path" to try as a workaround.
