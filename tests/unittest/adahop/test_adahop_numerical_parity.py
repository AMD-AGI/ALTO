# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Phase-1 numerical parity test: AdaHOP per-slot modes vs plain MXFP4.

Motivation
----------
On identical MI350X hardware, ``gpt_oss_20b_adahop`` trains WORSE than plain
``gpt_oss_20b_lpt`` even though AdaHOP is supposed to be a strict superset
(Hadamard incoherence processing + outlier extraction). The validation-loss
gap *widens* over training, which points at biased gradients rather than a
one-off noisier forward.

This test isolates the numerical behaviour of the two autograd paths for a
single Linear, so we can see *which output* (``y`` / ``grad_x`` / ``grad_w``)
and *which AdaHOP mode* diverges from the bf16 reference more than the plain
MXFP4 baseline does.

Suspects being probed (see plan keen-questing-ember):
  * S1 — AdaHOP ``hadamard`` mode drops 2D-block weight scaling (forward/y).
  * S2 — ``inner_outlier_extract_right`` (dominant backward_gw mode) quantizes
         grad_output with NO stochastic rounding → biased grad_w.
  * S4 — ``inner_outlier_extract_left`` (backward_gx) SR propagation.

Interpretation
--------------
For an *unbiased* quantizer, averaging over many random gradients drives the
mean relative error of grad_w toward ~0 (the ``_bias`` metrics below). A
quantizer that silently drops stochastic rounding will show a grad_w mean-bias
that does NOT shrink with averaging and is materially larger than the plain
MXFP4 baseline's. That is the fingerprint of S2.

Requires a real GPU (MI350X/CDNA4 for the fused kernels; CDNA3 exercises the
QDQ fallbacks). Skips on CPU-only boxes.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("AdaHOP numerical parity test requires a GPU", allow_module_level=True)

try:
    from alto.kernels.fp4.mxfp4.mxfp_linear import MXFP4LinearFunction
    from alto.modifiers.lpt.adahop_internals.mxfp4_linear_function import (
        MXFP4AdaHOPLinearFunction,
    )
    from alto._adahop_bridge import HadamardFactory
except RuntimeError as exc:  # triton driver init can fail even with a GPU present
    pytest.skip(f"kernel import failed: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shapes divisible by 32 (MXFP4 block) and >= OUTLIER_K=64 on the extracted
# axis so outlier extraction has room. gpt_oss_20b wq is 2880 -> 4096.
M, K, N = 2048, 2880, 4096
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _rel_err(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """Frobenius relative error ||approx - ref|| / ||ref||."""
    approx = approx.float()
    ref = ref.float()
    return (torch.linalg.vector_norm(approx - ref) /
            torch.linalg.vector_norm(ref).clamp_min(1e-12)).item()


def _mean_bias(approx: torch.Tensor, ref: torch.Tensor) -> float:
    """Normalized mean signed error — detects a *systematic* offset (bias)
    that stochastic rounding is supposed to cancel. Near 0 = unbiased."""
    approx = approx.float()
    ref = ref.float()
    return ((approx - ref).mean() / ref.abs().mean().clamp_min(1e-12)).item()


def _make_inputs(seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(M, K, generator=g, device=DEVICE, dtype=DTYPE)
    w = torch.randn(N, K, generator=g, device=DEVICE, dtype=DTYPE) * 0.02
    # Inject genuine column outliers into x so the "col" pattern / outlier
    # extraction paths have something real to act on (mirrors what the
    # calibration classifier detected on the real run: x=col on most layers).
    x[:, ::128] *= 25.0
    grad_y = torch.randn(M, N, generator=g, device=DEVICE, dtype=DTYPE)
    return x, w, grad_y


def _reference(x, w, grad_y):
    """bf16 ground truth for y, grad_x, grad_w."""
    xr = x.detach().float().requires_grad_(True)
    wr = w.detach().float().requires_grad_(True)
    y = xr @ wr.T
    y.backward(grad_y.float())
    return y.detach(), xr.grad.detach(), wr.grad.detach()


def _run_baseline(x, w, grad_y):
    """Plain MXFP4 path with the recipe's flags (2dblock_w, hadamard, sr_grad)."""
    xin = x.detach().clone().requires_grad_(True)
    win = w.detach().clone().requires_grad_(True)
    with torch.no_grad():
        ht = HadamardFactory.create_transform(device=win.device)
    y = MXFP4LinearFunction.apply(
        xin, win,
        False,  # use_2dblock_x
        True,   # use_2dblock_w
        True,   # use_sr_grad
        False,  # use_dge
        "none",  # clip_mode
        False,  # use_macro_block_scaling
        ht,
        None,   # module_id
    )
    y.backward(grad_y)
    return y.detach(), xin.grad.detach(), win.grad.detach()


def _run_adahop(x, w, grad_y, fy, gx, gw, use_sr_grad=True):
    """AdaHOP path with explicit per-slot modes and SR toggle."""
    xin = x.detach().clone().requires_grad_(True)
    win = w.detach().clone().requires_grad_(True)
    with torch.no_grad():
        ht = HadamardFactory.create_transform(device=win.device)
    y = MXFP4AdaHOPLinearFunction.apply(
        xin, win,
        use_sr_grad,
        ht,
        fy, gx, gw,
    )
    y.backward(grad_y)
    return y.detach(), xin.grad.detach(), win.grad.detach()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adahop_none_matches_baseline():
    """Sanity: AdaHOP with all slots 'none' must track plain MXFP4 closely.
    A large gap here means the AdaHOP autograd Function has a plumbing bug
    independent of any transform (rules S1/S2 in vs out)."""
    x, w, grad_y = _make_inputs(seed=0)
    y_ref, gx_ref, gw_ref = _reference(x, w, grad_y)

    y_b, gx_b, gw_b = _run_baseline(x, w, grad_y)
    y_a, gx_a, gw_a = _run_adahop(x, w, grad_y, "none", "none", "none")

    for name, b, a in (("y", y_b, y_a), ("grad_x", gx_b, gx_a), ("grad_w", gw_b, gw_a)):
        eb, ea = _rel_err(b, y_ref if name == "y" else (gx_ref if name == "grad_x" else gw_ref)), \
                 _rel_err(a, y_ref if name == "y" else (gx_ref if name == "grad_x" else gw_ref))
        print(f"[none]   {name:7s} baseline_relerr={eb:.4f} adahop_relerr={ea:.4f}")
        # AdaHOP 'none' should not be materially worse than baseline.
        assert ea <= eb * 1.5 + 0.02, (
            f"AdaHOP none-mode {name} rel-err {ea:.4f} >> baseline {eb:.4f}: "
            "plumbing bug independent of transforms")


def test_grad_w_bias_across_modes():
    """Core S2 probe. Averages grad_w over many random gradients and reports
    the *mean signed bias* per backward_gw mode. Unbiased quantization (SR
    on) → bias shrinks toward 0. If 'inner_outlier_extract_right' shows a
    persistent bias much larger than the baseline / 'hadamard' modes, SR was
    silently dropped on that path (S2)."""
    n_avg = 16
    modes = ["hadamard", "inner_outlier_extract_right", "full_precision"]

    baseline_bias = 0.0
    acc_bias = {m: 0.0 for m in modes}
    acc_relerr = {m: 0.0 for m in modes}
    baseline_relerr = 0.0

    for i in range(n_avg):
        x, w, grad_y = _make_inputs(seed=100 + i)
        _, _, gw_ref = _reference(x, w, grad_y)

        _, _, gw_b = _run_baseline(x, w, grad_y)
        baseline_bias += _mean_bias(gw_b, gw_ref) / n_avg
        baseline_relerr += _rel_err(gw_b, gw_ref) / n_avg

        for m in modes:
            # Keep the forward + gx legs on 'hadamard' so we isolate backward_gw.
            _, _, gw_a = _run_adahop(x, w, grad_y, "hadamard", "hadamard", m)
            acc_bias[m] += _mean_bias(gw_a, gw_ref) / n_avg
            acc_relerr[m] += _rel_err(gw_a, gw_ref) / n_avg

    print(f"\n[grad_w bias over {n_avg} draws]")
    print(f"  baseline(mxfp4,sr)        bias={baseline_bias:+.5f} relerr={baseline_relerr:.4f}")
    for m in modes:
        print(f"  gw={m:28s} bias={acc_bias[m]:+.5f} relerr={acc_relerr[m]:.4f}")

    # The dominant real-run mode. If its |bias| is an order of magnitude worse
    # than baseline, that's the widening-gap culprit (S2).
    ioe = abs(acc_bias["inner_outlier_extract_right"])
    base = abs(baseline_bias)
    print(f"  => inner_outlier_extract_right |bias|={ioe:.5f} vs baseline |bias|={base:.5f}")
    # Not a hard assert on the ratio (we want the number reported even when it
    # passes); flag only an egregious systematic bias.
    assert ioe < 0.05, (
        f"inner_outlier_extract_right grad_w has a large systematic bias "
        f"({ioe:.5f}); stochastic rounding likely dropped on this path (S2)")


def test_forward_y_hadamard_vs_baseline():
    """S1 probe: compare forward-y rel-err of AdaHOP 'hadamard' mode against
    the baseline (which keeps 2D-block weight scaling). AdaHOP should be
    <= baseline; if it's worse, the 1D-only iht_quantization is a real
    forward precision downgrade."""
    x, w, grad_y = _make_inputs(seed=7)
    y_ref, _, _ = _reference(x, w, grad_y)

    y_b, _, _ = _run_baseline(x, w, grad_y)
    y_a, _, _ = _run_adahop(x, w, grad_y, "hadamard", "hadamard", "hadamard")

    eb = _rel_err(y_b, y_ref)
    ea = _rel_err(y_a, y_ref)
    print(f"\n[forward y] baseline_relerr={eb:.4f} adahop_hadamard_relerr={ea:.4f}")
    assert ea <= eb * 1.5 + 0.02, (
        f"AdaHOP hadamard forward-y rel-err {ea:.4f} >> baseline {eb:.4f} (S1)")


def _which_extract_branch():
    """Report which inner_outlier_extract branch this GPU runs, so the test
    output states what was actually verified (CDNA3 QDQ vs CDNA4 fused)."""
    try:
        from alto.kernels.fp4.mxfp4.mxfp_quantization import is_cdna4
        return "CDNA4 (fused/dot_scaled)" if is_cdna4() else "CDNA3 (pytorch QDQ)"
    except Exception:
        return "unknown"


def test_sr_reduces_grad_w_bias_outlier_right():
    """The SR fix: threading use_sr into inner_outlier_extract_right's grad
    quantization should make grad_w UNBIASED (mean signed error -> 0 with
    averaging). With SR off (the pre-fix behavior), grad_w carries a persistent
    systematic bias. This directly verifies the fix on whatever branch this GPU
    runs (CDNA3 pytorch-QDQ or CDNA4 fused) — both were patched.

    backward_gw = inner_outlier_extract_right is the dominant real-run mode
    (86/96 layers in the calibrated run), so this is the path that matters.
    """
    branch = _which_extract_branch()
    n_avg = 32
    bias_sr_on = 0.0
    bias_sr_off = 0.0
    relerr_sr_on = 0.0
    for i in range(n_avg):
        x, w, grad_y = _make_inputs(seed=500 + i)
        _, _, gw_ref = _reference(x, w, grad_y)
        # Isolate backward_gw = inner_outlier_extract_right; keep fwd + gx on hadamard.
        _, _, gw_on = _run_adahop(x, w, grad_y, "hadamard", "hadamard",
                                  "inner_outlier_extract_right", use_sr_grad=True)
        _, _, gw_off = _run_adahop(x, w, grad_y, "hadamard", "hadamard",
                                   "inner_outlier_extract_right", use_sr_grad=False)
        bias_sr_on += _mean_bias(gw_on, gw_ref) / n_avg
        bias_sr_off += _mean_bias(gw_off, gw_ref) / n_avg
        relerr_sr_on += _rel_err(gw_on, gw_ref) / n_avg

    print(f"\n[SR fix / grad_w, backward_gw=inner_outlier_extract_right, {branch}]")
    print(f"  SR on : mean_bias={bias_sr_on:+.6f}  relerr={relerr_sr_on:.4f}")
    print(f"  SR off: mean_bias={bias_sr_off:+.6f}")
    print(f"  => |bias| SR on={abs(bias_sr_on):.6f}  SR off={abs(bias_sr_off):.6f}")

    # The SR fix must actually change the result (proves use_sr is threaded through
    # to the kernel's grad quantization on this branch) AND make it less biased.
    assert abs(bias_sr_on) != abs(bias_sr_off), (
        "SR on/off produced identical grad_w bias -> use_sr is NOT reaching the "
        f"outlier kernel's grad quant on {branch}")
    assert abs(bias_sr_on) <= abs(bias_sr_off) + 1e-6, (
        f"SR did not reduce grad_w bias on {branch}: "
        f"|bias| on={abs(bias_sr_on):.6f} off={abs(bias_sr_off):.6f}")
