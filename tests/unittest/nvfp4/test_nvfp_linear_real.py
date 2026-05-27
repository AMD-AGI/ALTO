# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Real-data op-level SNR tests for NVFP4 linear, paired with MXFP4.

This is the consumer side of the workflow described in
``alto/kernels/fp4/real_data_utils.py`` and
``scripts/dump_real_test_activations.py``.

Why this file is necessary on top of ``test_nvfp_linear.py``
============================================================
The synthetic ``test_nvfp4_linear_autograd_function`` uses
``prepare_data(..., pattern="random")`` (Gaussian + 0.5 % sparse outliers).
That distribution is convenient for CI but it understates two things
that matter for op-level FP4 quality:

* Real transformer activations have much heavier tails than Gaussian,
  which stresses the per-block ``amax`` scale encoding (E8M0 for MX
  vs E4M3 for NV) very differently from the synthetic baseline.
* The current synthetic baseline already shows NVFP4 inner-only +
  large K + SR collapsing to ~2 dB dX (see ``nvfp4_mxfp4_op_level_test.md``
  §3.2).  Whether that collapse is a synthetic artefact or a real
  property of NVFP4 on production activations is the headline open
  question this test exists to answer.

So this file feeds **the same real ``(x, w)``** -- captured from a
GPT-OSS-20B BF16 forward -- through both NVFP4 and MXFP4, and reports
SNR for forward output ``O``, activation gradient ``dX``, and weight
gradient ``dW`` against a BF16 reference computed on the same inputs.

What is NOT in this starter file (and why)
==========================================
* Only the **dense linear** layers (``dense_early/mid/late`` ->
  ``layers.{0,12,22}.attention.wq``) are exercised here.  Grouped
  GEMM bundles live in a sibling file ``test_nvfp_grouped_gemm_real.py``
  (planned PR-B follow-up) so the two op shapes can ship and
  iterate independently.
* Only the ``minimal`` recipe (1D/1D, no SR, no outer) and the
  ``production_nv_outer`` recipe are wired here.  The MXFP4-only
  ``production_mx`` recipe is asserted from the MXFP4 mirror test.
* SNR floors are deliberately **loose** in this first version (5 dB
  hard / 7 dB aggregate).  They are a "did the op blow up?" guardrail,
  not the production threshold.  After the first end-to-end run on the
  healthy node we will tighten them based on the *measured* per-recipe
  distribution -- see ``nvfp4_mxfp4_op_level_test.md`` for the
  synthetic baseline thresholds we will refer back to.

Test design notes
=================
* The fixture returns a bundle dict; this file pulls out ``x``, ``w``,
  and ``dy`` (preferring ``dy_real`` when ``--with-backward`` was used,
  otherwise falling back to ``dy_normal_std_y``).  Synthesised dy is
  Gaussian with the same std as ``y`` -- close to a real upstream
  gradient in scale and uncorrelated with the rest of the chain, which
  is exactly what we want for isolating the NV / MX kernel difference.
* For each test case we do **three independent forward+backward passes**
  -- one BF16 reference, one NVFP4, one MXFP4 -- starting from a fresh
  ``.requires_grad_(True)`` clone of x / w each time.  This guarantees
  the only differences in dX / dW come from the FP4 quantization paths,
  not from autograd state leaking between runs.
* Each test prints a ``[REAL_DATA_SNR]`` line that the planned
  ``scripts/aggregate_real_data_snr.py`` parses; do **not** rename or
  drop this prefix without updating that script.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tabulate import tabulate

from alto.kernels.fp4.real_data_utils import (
    DENSE_TAGS,
    KEY_DY_NORMAL_STD_Y,
    KEY_DY_REAL,
    KEY_W,
    KEY_X,
)
from alto.kernels.fp4.testing_utils import calc_snr, calc_cossim
from alto.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction
from alto.kernels.fp4.mxfp4.mxfp_linear import MXFP4LinearFunction


# ---------------------------------------------------------------------------
# Recipe descriptors
# ---------------------------------------------------------------------------
#
# Each entry maps to the constructor knobs for the two LinearFunction
# autograd classes.  ``nv`` / ``mx`` may be None to indicate "this recipe
# does not have a counterpart in the other format" -- the runner then
# skips that side of the comparison.  Today both production recipes are
# format-specific, which is the right semantics: NV's production default
# is outer-scale + 2D-w + SR; MX's production default is Hadamard + 2D-w
# + SR.  Comparing the two head-to-head only makes sense for the
# ``minimal`` recipe.

_RECIPES: dict[str, dict] = {
    "minimal": {
        "nv": dict(
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False, use_outer_scale=False,
        ),
        "mx": dict(
            use_2dblock_x=False, use_2dblock_w=False,
            use_sr_grad=False, use_dge=False,
            clip_mode="none", use_macro_block_scaling=False,
            hadamard_transform=None,
        ),
    },
    "production_nv_outer": {
        "nv": dict(
            use_2dblock_x=False, use_2dblock_w=True,
            use_sr_grad=True, use_outer_scale=True,
        ),
        "mx": None,
    },
}


# Loose first-pass floors.  Real data heavy-tail can drop NV / MX
# per-tensor SNR a few dB below the synthetic baseline, especially on
# the inner-only ``minimal`` recipe.  Tightening these requires one
# end-to-end measured run -- do not raise them blindly.
_REAL_DATA_SNR_FLOORS: dict[str, dict[str, float]] = {
    # recipe -> {hard, aggregate}
    "minimal":             {"hard": 5.0, "aggregate": 7.0},
    "production_nv_outer": {"hard": 8.0, "aggregate": 10.0},
}


def _pick_dy(bundle: dict) -> tuple[torch.Tensor, str]:
    """Return the dy to use for backward + a label for the metrics line.

    Real dy from a captured backward pass is strictly preferred when
    available (``--with-backward`` on the dump), otherwise we fall back
    to ``N(0, std(y))`` which has the right magnitude scale for the
    layer in question without depending on backward instrumentation.
    """
    if KEY_DY_REAL in bundle:
        return bundle[KEY_DY_REAL], "real"
    return bundle[KEY_DY_NORMAL_STD_Y], "normal_std_y"


def _bf16_reference(x: torch.Tensor, w: torch.Tensor, dy: torch.Tensor):
    """Reference forward + custom-dy backward in BF16.

    We use ``F.linear`` (no bias) here even when the captured module was
    ``nn.Linear(bias=True)``: bias is not quantised by NVFP4 / MXFP4
    linear paths and would otherwise show up as a constant offset on
    forward SNR.  This is the same convention used in
    ``test_nvfp4_linear_autograd_function``.
    """
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)
    y_ref = F.linear(x_ref, w_ref)
    y_ref.backward(dy)
    return y_ref.detach(), x_ref.grad.detach(), w_ref.grad.detach()


def _nv_forward_backward(x, w, dy, knobs):
    x_nv = x.detach().clone().requires_grad_(True)
    w_nv = w.detach().clone().requires_grad_(True)
    y = NVFP4LinearFunction.apply(x_nv, w_nv, **knobs)
    y.backward(dy)
    return y.detach(), x_nv.grad.detach(), w_nv.grad.detach()


def _mx_forward_backward(x, w, dy, knobs):
    x_mx = x.detach().clone().requires_grad_(True)
    w_mx = w.detach().clone().requires_grad_(True)
    y = MXFP4LinearFunction.apply(x_mx, w_mx, **knobs)
    y.backward(dy)
    return y.detach(), x_mx.grad.detach(), w_mx.grad.detach()


def _check_floor(name: str, snrs: dict[str, float], floors: dict[str, float], ctx: str):
    """Assert per-tensor hard floor + median/mean aggregate floor.

    Modelled after :func:`alto.kernels.fp4.testing_utils.check_nvfp4_autograd_snr`
    but with a recipe-keyed real-data floor table.  Failure messages
    include the SNR values and context so aggregator output (and
    on-call triage) is self-contained.
    """
    hard, aggregate = floors["hard"], floors["aggregate"]
    for tname, val in snrs.items():
        assert val > hard, (
            f"{ctx} {name}/{tname} SNR {val:.2f} dB < hard {hard:.2f} dB "
            f"(snrs={snrs})"
        )
    vals = list(snrs.values())
    med = sorted(vals)[len(vals) // 2]
    mean = sum(vals) / len(vals)
    assert med > aggregate, (
        f"{ctx} {name} median SNR {med:.2f} dB < aggregate {aggregate:.2f} dB (snrs={snrs})"
    )
    assert mean > aggregate, (
        f"{ctx} {name} mean SNR {mean:.2f} dB < aggregate {aggregate:.2f} dB (snrs={snrs})"
    )


# ---------------------------------------------------------------------------
# Main parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.real_data
@pytest.mark.parametrize("real_data", list(DENSE_TAGS), indirect=True)
@pytest.mark.parametrize("recipe", list(_RECIPES))
def test_nvfp4_vs_mxfp4_linear_real(real_data, recipe):
    """NVFP4 vs MXFP4 op-level SNR on a real GPT-OSS-20B dense layer.

    Always runs the NVFP4 path; runs the MXFP4 path only for recipes
    that have an MX counterpart.  Forward and gradient SNR are reported
    against a BF16 reference computed on the *same* (x, w, dy) triplet.
    """
    bundle = real_data
    tag = bundle["_loader"]["tag"]

    x = bundle[KEY_X].to(dtype=torch.bfloat16, device="cuda")
    w = bundle[KEY_W].to(dtype=torch.bfloat16, device="cuda")
    dy, dy_label = _pick_dy(bundle)
    dy = dy.to(dtype=torch.bfloat16, device="cuda")

    cfg = _RECIPES[recipe]

    # BF16 reference (computed once, shared by both formats).
    y_ref, dx_ref, dw_ref = _bf16_reference(x, w, dy)

    rows = []

    # NVFP4 leg -- always present.
    y_nv, dx_nv, dw_nv = _nv_forward_backward(x, w, dy, cfg["nv"])
    nv_snrs = {
        "O":  calc_snr(y_ref,  y_nv),
        "dX": calc_snr(dx_ref, dx_nv),
        "dW": calc_snr(dw_ref, dw_nv),
    }
    rows.append([
        "NV", f"{nv_snrs['O']:.2f}", f"{nv_snrs['dX']:.2f}", f"{nv_snrs['dW']:.2f}",
        f"{calc_cossim(y_ref, y_nv):.4f}",
    ])

    # MXFP4 leg -- only when the recipe has an MX counterpart.
    mx_snrs: dict[str, float] | None = None
    if cfg["mx"] is not None:
        y_mx, dx_mx, dw_mx = _mx_forward_backward(x, w, dy, cfg["mx"])
        mx_snrs = {
            "O":  calc_snr(y_ref,  y_mx),
            "dX": calc_snr(dx_ref, dx_mx),
            "dW": calc_snr(dw_ref, dw_mx),
        }
        rows.append([
            "MX", f"{mx_snrs['O']:.2f}", f"{mx_snrs['dX']:.2f}", f"{mx_snrs['dW']:.2f}",
            f"{calc_cossim(y_ref, y_mx):.4f}",
        ])

    # Print a tabular summary AND a single grep-friendly line per
    # (recipe, tag, format) for the aggregator to parse.  Keep both:
    # tabulate is for the human reading the pytest log, the prefixed
    # line is for the aggregator script.
    print()
    print(f"recipe={recipe}  layer={tag}  dy={dy_label}  "
          f"x_shape={list(x.shape)}  w_shape={list(w.shape)}")
    print(tabulate(rows, headers=["fmt", "O", "dX", "dW", "CosSim(O)"],
                   tablefmt="github"))
    print(f"[REAL_DATA_SNR] recipe={recipe} layer={tag} fmt=NV "
          f"O={nv_snrs['O']:.2f} dX={nv_snrs['dX']:.2f} dW={nv_snrs['dW']:.2f} dy={dy_label}")
    if mx_snrs is not None:
        print(f"[REAL_DATA_SNR] recipe={recipe} layer={tag} fmt=MX "
              f"O={mx_snrs['O']:.2f} dX={mx_snrs['dX']:.2f} dW={mx_snrs['dW']:.2f} dy={dy_label}")

    # Enforce loose v1 floors per format.  These are intentionally
    # generous; see module docstring for tightening plan.
    floors = _REAL_DATA_SNR_FLOORS[recipe]
    _check_floor("NV", nv_snrs, floors, ctx=f"[{recipe}/{tag}]")
    if mx_snrs is not None:
        _check_floor("MX", mx_snrs, floors, ctx=f"[{recipe}/{tag}]")
