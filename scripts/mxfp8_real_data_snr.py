#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Op-level SNR / CosSim: MXFP8-e4m3 on REAL GPT-OSS-20B activations.

Mirrors the methodology of the NVFP4/MXFP4/AMD-FP4 real-data SNR report
(amdfp4_vs_nvfp4_mxfp4_real_data_snr_report_zh.md), but measures MXFP8-e4m3
instead.  Reuses the same real-activation bundles — no new data collection needed.

  Data:   /wekafs/zhitwang/test_data/real_activations/step_latest_20260603_035647
  Layers: L1 (early), L12 (mid), L20 (late)
  Ops:    attention wq (dense linear) — forward O + backward dX / dW
          MoE experts mlp1 (grouped GEMM) — forward O

Two recipe tiers, mirroring the FP4 report structure:

  plain  — all anti-outlier mechanisms off (use_2dblock_w=False, sr_grad=False).
           Isolates raw MXFP8-e4m3 format quality.  MXFP8 has no hadamard /
           outer_scale, so plain == deploy minus use_2dblock_w.

  deploy — production-aligned ("能开的开关就开").
           use_2dblock_w=True (can enable on CDNA3).
           use_sr_grad=False  (CDNA3 does NOT support the SR ASM path; deploy
                               recipe lpt_recipe_mxfp8.yaml also sets False).
           use_hadamard / two_level_scaling: not applicable to MXFP8.

FP4 comparison columns use the corresponding tier from the FP4 report:
  plain  -> FP4 plain  (sr_grad=False, no 2dblock_w/hadamard/outer_scale)
  deploy -> FP4 deploy (sr_grad=True,  2dblock_w+hadamard+outer_scale enabled)

SNR  = 10 * log10(||ref||^2 / ||ref - q||^2)  dB, higher is better
CosSim = <ref, q> / (||ref|| * ||q||)          1.0 = perfect

Run (single GPU):
  HIP_VISIBLE_DEVICES=0 HSA_NO_SCRATCH_RECLAIM=1 TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1 \\
  PYTHONPATH=/wekafs/yuesun/workspace/repos/ALTO \\
  python scripts/mxfp8_real_data_snr.py
"""

import os
import torch

ROOT = os.environ.get(
    "ALTO_REAL_DATA_DIR",
    "/wekafs/zhitwang/test_data/real_activations/step_latest_20260603_035647",
)

DENSE   = [("L1",  "dense_early"),
           ("L12", "dense_mid"),
           ("L20", "dense_late")]
MOE     = [("L1",  "grouped_early"),
           ("L12", "grouped_mid"),
           ("L20", "grouped_late")]

FP8_VARIANT = "e4m3"

# ---------------------------------------------------------------------------
# Recipe configs
# ---------------------------------------------------------------------------

# plain: all anti-outlier mechanisms off — mirrors FP4 plain tier.
PLAIN_CFG = dict(use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=False)

# deploy: production-aligned ("能开的开关就开").
#   use_2dblock_w=True  — supported on CDNA3, matches lpt_recipe_mxfp8.yaml.
#   use_sr_grad=False   — CDNA3 does not support SR ASM; recipe also False.
#   hadamard / outer_scale: not applicable to MXFP8.
DEPLOY_CFG = dict(use_2dblock_x=False, use_2dblock_w=True, use_sr_grad=False)

# ---------------------------------------------------------------------------
# FP4 reference numbers (from amdfp4_vs_nvfp4_mxfp4_real_data_snr_report_zh.md)
# ---------------------------------------------------------------------------

# §2.1 Plain (sr_grad=False, no 2dblock_w/hadamard/outer_scale)
# Format: {layer: (NV_O, MX_O, AMD_O, NV_dX, MX_dX, AMD_dX, NV_dW, MX_dW, AMD_dW)}
_FP4_PLAIN_DENSE = {
    "L1":  (16.34, 14.23, 17.71,   13.75, 13.14, 14.45,   20.53, 18.26, 20.55),
    "L12": (14.10, 15.92, 19.44,    9.84, 13.57, 15.68,   21.95, 20.88, 24.81),
    "L20": (14.36, 16.92, 20.53,    6.12,  7.63,  9.95,   29.18, 22.72, 29.78),
}
# Format: {layer: (NV, MX, AMD)}
_FP4_PLAIN_MOE = {
    "L1":  (8.07, 8.89,  9.12),
    "L12": (6.00, 11.32, 11.68),
    "L20": (6.10, 10.79, 11.23),
}

# §2.2 Deploy (sr_grad=True, 2dblock_w+hadamard+outer_scale all enabled)
# Format: {layer: (NV_O, MX_O, AMD_O, NV_dX, MX_dX, AMD_dX, NV_dW, MX_dW, AMD_dW)}
_FP4_DEPLOY_DENSE = {
    "L1":  (15.28, 11.68, 15.28,   12.46, 10.22, 12.08,   17.31, 15.82, 18.40),
    "L12": (17.67, 14.10, 17.67,   12.11,  9.65, 12.39,   23.53, 18.66, 24.31),
    "L20": (17.94, 13.87, 17.94,    6.47,  3.25,  6.24,   29.42, 19.60, 31.22),
}
# Format: {layer: (NV, MX, AMD)}
_FP4_DEPLOY_MOE = {
    "L1":  (8.98, 8.59,  8.98),
    "L12": (11.32, 10.51, 11.32),
    "L20": (10.56,  9.08, 10.56),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(slot: str) -> dict:
    path = f"{ROOT}/{slot}/bundle.pt"
    return torch.load(path, map_location="cuda", weights_only=False)


def _calc_snr(ref: torch.Tensor, q: torch.Tensor) -> float:
    """SNR = 10*log10(||ref||^2 / ||ref-q||^2) dB."""
    from alto.kernels.fp4.testing_utils import calc_snr
    return calc_snr(ref, q)


def _calc_cossim(ref: torch.Tensor, q: torch.Tensor) -> float:
    """Cosine similarity, computed in float32."""
    from alto.kernels.fp4.testing_utils import calc_cossim
    return calc_cossim(ref, q)


# ---------------------------------------------------------------------------
# Dense linear (attention wq): forward O + backward dX, dW
# ---------------------------------------------------------------------------

def dense_snr(slot: str, use_2dblock_x: bool, use_2dblock_w: bool,
              use_sr_grad: bool) -> dict:
    """Return SNR and CosSim for O / dX / dW on one dense-linear bundle."""
    from alto.kernels.mxfp8.mxfp8_linear import _to_mxfp8_then_scaled_mm

    d = _load(slot)
    x  = d["x"].reshape(-1, d["x"].shape[-1])          # [S, K]
    w  = d["w"]                                          # [N, K]
    dy = d["dy"].reshape(-1, d["w"].shape[0]).float()   # [S, N]  FP32 for stable grad

    # BF16 reference (analytic gradients in FP32)
    y_ref  = x.float() @ w.float().T                    # [S, N]
    dX_ref = dy @ w.float()                             # [S, K]
    dW_ref = dy.T @ x.float()                           # [N, K]

    # MXFP8 forward + backward via autograd
    xq = x.clone().requires_grad_(True)
    wq = w.clone().requires_grad_(True)
    y = _to_mxfp8_then_scaled_mm(
        xq, wq,
        fp8_variant   = FP8_VARIANT,
        use_sr_grad   = use_sr_grad,
        use_2dblock_x = use_2dblock_x,
        use_2dblock_w = use_2dblock_w,
    )
    y.backward(d["dy"].reshape(-1, d["w"].shape[0]))

    return {
        "O_snr":    _calc_snr(y_ref,  y.detach()),
        "O_cos":    _calc_cossim(y_ref,  y.detach()),
        "dX_snr":   _calc_snr(dX_ref, xq.grad),
        "dX_cos":   _calc_cossim(dX_ref, xq.grad),
        "dW_snr":   _calc_snr(dW_ref, wq.grad),
        "dW_cos":   _calc_cossim(dW_ref, wq.grad),
    }


# ---------------------------------------------------------------------------
# MoE grouped experts (mlp1): forward O
# ---------------------------------------------------------------------------

def _pad_to_pow2(t: torch.Tensor) -> tuple:
    """Pad dim-0 to the next power of 2 (Triton arange constraint). Returns (padded, orig_len)."""
    m = t.shape[0]
    p = 1
    while p < m:
        p *= 2
    if p == m:
        return t, m
    pad = torch.zeros(p - m, *t.shape[1:], dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=0), m


def moe_snr(slot: str, use_2dblock_x: bool, use_2dblock_w: bool) -> dict:
    """Return SNR and CosSim for grouped-GEMM forward on one MoE bundle.

    Uses per-expert QDQ + BF16 matmul (same methodology as the FP4 report).
    Pads each expert's token slice to a power of 2 before calling convert_to_mxfp8
    (Triton arange constraint), then trims back before the matmul.
    """
    from alto.kernels.mxfp8.mxfp8_quantization import (
        convert_to_mxfp8,
        convert_from_mxfp8,
    )

    d    = _load(slot)
    x    = d["x"]                              # [total_tokens, K]
    w    = d["w"]                              # [E, N, K]
    ntpe = d["num_tokens_per_expert"].tolist() # [E]

    refs, qs = [], []
    offset = 0
    for e, n_tok in enumerate(ntpe):
        if n_tok == 0:
            offset += n_tok
            continue

        xe = x[offset : offset + n_tok]  # [n_tok, K]
        we = w[e]                         # [N, K]
        offset += n_tok

        # Pad to power-of-2 rows, quantize, then trim back
        xe_pad, orig = _pad_to_pow2(xe)
        xe_lp, xe_s = convert_to_mxfp8(xe_pad, mxfp_format=FP8_VARIANT,
                                        axis=-1, is_2d_block=use_2dblock_x)
        xe_dq = convert_from_mxfp8(xe_lp, xe_s, output_dtype=torch.float32,
                                    axis=-1, is_2d_block=use_2dblock_x)
        xe_dq = xe_dq[:orig]  # trim padding rows

        # Weight: N is always from model config (power-of-2 safe); K also fine
        we_lp, we_s = convert_to_mxfp8(we, mxfp_format=FP8_VARIANT,
                                        axis=-1, is_2d_block=use_2dblock_w)
        we_dq = convert_from_mxfp8(we_lp, we_s, output_dtype=torch.float32,
                                    axis=-1, is_2d_block=use_2dblock_w)

        refs.append(xe.float() @ we.float().T)   # [n_tok, N]
        qs.append(xe_dq @ we_dq.T)               # [n_tok, N]

    ref_cat = torch.cat(refs)
    q_cat   = torch.cat(qs)
    return {
        "O_snr": _calc_snr(ref_cat, q_cat),
        "O_cos": _calc_cossim(ref_cat, q_cat),
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _fmt(snr: float, cos: float) -> str:
    return f"{snr:7.2f} ({cos:.4f})"


def _print_dense_table(results: list) -> None:
    hdr = (f"{'layer':5}  "
           f"{'O  SNR(CosSim)':>22}  "
           f"{'dX SNR(CosSim)':>22}  "
           f"{'dW SNR(CosSim)':>22}")
    print(hdr)
    print("-" * len(hdr))
    for lbl, r in results:
        o  = _fmt(r["O_snr"],  r["O_cos"])
        dx = _fmt(r["dX_snr"], r["dX_cos"])
        dw = _fmt(r["dW_snr"], r["dW_cos"])
        print(f"{lbl:5}  {o:>22}  {dx:>22}  {dw:>22}")


def _print_moe_table(results: list) -> None:
    hdr = f"{'layer':5}  {'O  SNR(CosSim)':>22}"
    print(hdr)
    print("-" * len(hdr))
    for lbl, r in results:
        print(f"{lbl:5}  {_fmt(r['O_snr'], r['O_cos']):>22}")


def _print_comparison_table(tier: str,
                             dense_by_lbl: dict,
                             moe_by_lbl: dict,
                             fp4_dense: dict,
                             fp4_moe: dict,
                             fp4_sr_note: str) -> None:
    """Side-by-side table: MXFP8 vs FP4 formats for one tier."""
    print()
    print("=" * 108)
    print(f"  COMPARISON [{tier.upper()}]: MXFP8-e4m3 vs FP4 formats")
    print(f"  FP4 source : amdfp4_vs_nvfp4_mxfp4_real_data_snr_report_zh.md §2 ({fp4_sr_note})")
    print("=" * 108)

    col = f"{'layer':5}  {'MXFP8':>8}  {'NVFP4':>8}  {'MXFP4':>8}  {'AMD-FP4':>9}  {'MX8-NV4':>9}  {'MX8-MX4':>9}"

    # Dense forward O
    print("\n--- Dense wq Forward O (SNR dB) ---")
    print(col)
    for lbl in ("L1", "L12", "L20"):
        mx8 = dense_by_lbl[lbl]["O_snr"]
        nv, mx4, amd = fp4_dense[lbl][0:3]
        print(f"{lbl:5}  {mx8:8.2f}  {nv:8.2f}  {mx4:8.2f}  {amd:9.2f}  {mx8-nv:+9.2f}  {mx8-mx4:+9.2f}")

    # Dense backward dX
    print(f"\n--- Dense wq Backward dX (SNR dB) ---")
    print(col)
    for lbl in ("L1", "L12", "L20"):
        mx8 = dense_by_lbl[lbl]["dX_snr"]
        nv, mx4, amd = fp4_dense[lbl][3:6]
        print(f"{lbl:5}  {mx8:8.2f}  {nv:8.2f}  {mx4:8.2f}  {amd:9.2f}  {mx8-nv:+9.2f}  {mx8-mx4:+9.2f}")

    # Dense backward dW
    print(f"\n--- Dense wq Backward dW (SNR dB) ---")
    print(col)
    for lbl in ("L1", "L12", "L20"):
        mx8 = dense_by_lbl[lbl]["dW_snr"]
        nv, mx4, amd = fp4_dense[lbl][6:9]
        print(f"{lbl:5}  {mx8:8.2f}  {nv:8.2f}  {mx4:8.2f}  {amd:9.2f}  {mx8-nv:+9.2f}  {mx8-mx4:+9.2f}")

    # MoE forward O
    print("\n--- MoE experts mlp1 Forward O (SNR dB) ---")
    print(col)
    for lbl in ("L1", "L12", "L20"):
        mx8 = moe_by_lbl[lbl]["O_snr"]
        nv, mx4, amd = fp4_moe[lbl]
        print(f"{lbl:5}  {mx8:8.2f}  {nv:8.2f}  {mx4:8.2f}  {amd:9.2f}  {mx8-nv:+9.2f}  {mx8-mx4:+9.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Data root : {ROOT}")
    print(f"Format    : MXFP8-{FP8_VARIANT}")
    print()

    # -----------------------------------------------------------------------
    # PLAIN tier
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("  [PLAIN] use_2dblock_w=False  use_sr_grad=False")
    print("  (mirrors FP4 plain: no 2dblock_w / hadamard / outer_scale / sr)")
    print("=" * 80)
    print()

    plain_dense_by_lbl: dict = {}
    plain_dense_rows = []
    for lbl, slot in DENSE:
        torch.manual_seed(0)
        r = dense_snr(slot, **PLAIN_CFG)
        plain_dense_by_lbl[lbl] = r
        plain_dense_rows.append((lbl, r))

    print("Dense (attention wq): O / dX / dW  SNR (dB)  CosSim")
    _print_dense_table(plain_dense_rows)

    plain_moe_by_lbl: dict = {}
    plain_moe_rows = []
    for lbl, slot in MOE:
        r = moe_snr(slot, use_2dblock_x=PLAIN_CFG["use_2dblock_x"],
                    use_2dblock_w=PLAIN_CFG["use_2dblock_w"])
        plain_moe_by_lbl[lbl] = r
        plain_moe_rows.append((lbl, r))

    print()
    print("MoE experts (mlp1): forward O  SNR (dB)  CosSim")
    _print_moe_table(plain_moe_rows)

    _print_comparison_table(
        "plain",
        plain_dense_by_lbl, plain_moe_by_lbl,
        _FP4_PLAIN_DENSE, _FP4_PLAIN_MOE,
        "plain, sr_grad=False, no 2dblock_w/hadamard/outer_scale",
    )

    # -----------------------------------------------------------------------
    # DEPLOY tier
    # -----------------------------------------------------------------------
    print()
    print()
    print("=" * 80)
    print("  [DEPLOY] use_2dblock_w=True  use_sr_grad=False")
    print("  (CDNA3: sr_grad=True not supported; all other available switches on)")
    print("=" * 80)
    print()

    deploy_dense_by_lbl: dict = {}
    deploy_dense_rows = []
    for lbl, slot in DENSE:
        torch.manual_seed(0)
        r = dense_snr(slot, **DEPLOY_CFG)
        deploy_dense_by_lbl[lbl] = r
        deploy_dense_rows.append((lbl, r))

    print("Dense (attention wq): O / dX / dW  SNR (dB)  CosSim")
    _print_dense_table(deploy_dense_rows)

    deploy_moe_by_lbl: dict = {}
    deploy_moe_rows = []
    for lbl, slot in MOE:
        r = moe_snr(slot, use_2dblock_x=DEPLOY_CFG["use_2dblock_x"],
                    use_2dblock_w=DEPLOY_CFG["use_2dblock_w"])
        deploy_moe_by_lbl[lbl] = r
        deploy_moe_rows.append((lbl, r))

    print()
    print("MoE experts (mlp1): forward O  SNR (dB)  CosSim")
    _print_moe_table(deploy_moe_rows)

    _print_comparison_table(
        "deploy",
        deploy_dense_by_lbl, deploy_moe_by_lbl,
        _FP4_DEPLOY_DENSE, _FP4_DEPLOY_MOE,
        "deploy, sr_grad=True, 2dblock_w+hadamard+outer_scale enabled",
    )


if __name__ == "__main__":
    main()
