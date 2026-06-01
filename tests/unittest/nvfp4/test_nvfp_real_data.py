# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""NVFP4 vs MXFP4 op-level SNR on REAL GPT-OSS-20B activations.

Companion to the synthetic comparison tests
(``test_nvfp4_linear_forward_compares_with_mxfp4_on_shared_cases`` and
``test_nvfp4_grouped_gemm_forward_compares_with_mxfp4``).  Those feed
``prepare_data`` (Gaussian + 0.5% sparse outliers) through the FP4 paths.
This file feeds *real* activations / trained weights captured from a live
GPT-OSS-20B NVFP4-test-2 run (see ``scripts/dump_real_test_activations.py``),
so we can check whether the synthetic conclusions — notably NVFP4's
forward advantage and its large-K backward disadvantage vs MXFP4 — hold on
the heavy-tailed distributions that actually occur during training.

All tests are marked ``real_data`` and auto-skip when the dumped bundles are
absent (default CI), so this file is inert unless you have run the dump.

The numbers printed under ``-s`` are the deliverable; the asserts are loose
floors that only catch catastrophic regressions (real activations are far
more outlier-heavy than synthetic data, so tight thresholds are not
meaningful here).
"""

import pytest
import torch
from tabulate import tabulate

from alto.kernels.fp4.nvfp4.nvfp_linear import (
    NVFP4LinearFunction,
    _to_nvfp4_then_scaled_mm,
)
from alto.kernels.fp4.mxfp4.mxfp_linear import (
    MXFP4LinearFunction,
    _to_mxfp4_then_scaled_mm,
)
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import nvfp4_grouped_gemm
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm import mxfp4_grouped_gemm
from .utils import calc_snr, calc_cossim


# Loose floors: real activations have heavier tails than synthetic data, so
# these only guard against catastrophic breakage, not recipe quality.
_FWD_SNR_FLOOR = 3.0
_BWD_SNR_FLOOR = 1.0

_DENSE_TAGS = ["dense_early", "dense_mid", "dense_late"]
_GROUPED_TAGS = ["grouped_early", "grouped_mid", "grouped_late"]


def _print_table(title, rows):
    print()
    print(f"# {title}")
    print(tabulate(rows, headers=["Tensor", "NVFP4 SNR", "MXFP4 SNR",
                                   "NV CosSim", "MX CosSim", "Δ(NV-MX) dB"],
                   tablefmt="github"))


# ---------------------------------------------------------------------------
# Dense linear — forward
# ---------------------------------------------------------------------------

@pytest.mark.real_data
@pytest.mark.parametrize("real_data", _DENSE_TAGS, indirect=True)
def test_real_dense_linear_forward(real_data):
    assert real_data["kind"] == "dense_linear"
    x = real_data["x"].to(torch.bfloat16)
    w = real_data["w"].to(torch.bfloat16)

    y_ref = torch.nn.functional.linear(x, w)
    y_nv = _to_nvfp4_then_scaled_mm(
        x, w, use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False, use_outer_scale=False,
    )
    y_mx = _to_mxfp4_then_scaled_mm(
        x, w, use_2dblock_x=False, use_2dblock_w=False,
        use_sr_grad=False, use_dge=False, clip_mode="none",
        use_hadamard=False, use_macro_block_scaling=False,
    )

    nv_snr, mx_snr = calc_snr(y_ref, y_nv), calc_snr(y_ref, y_mx)
    _print_table(
        f"real dense fwd [{real_data['tag']}] K={x.shape[-1]}",
        [["O", f"{nv_snr:.2f}", f"{mx_snr:.2f}",
          f"{calc_cossim(y_ref, y_nv):.4f}", f"{calc_cossim(y_ref, y_mx):.4f}",
          f"{nv_snr - mx_snr:+.2f}"]],
    )

    assert torch.isfinite(y_nv).all() and torch.isfinite(y_mx).all()
    assert nv_snr > _FWD_SNR_FLOOR, f"NVFP4 fwd SNR too low: {nv_snr:.2f}"
    assert mx_snr > _FWD_SNR_FLOOR, f"MXFP4 fwd SNR too low: {mx_snr:.2f}"


# ---------------------------------------------------------------------------
# Dense linear — backward (real upstream gradient injection)
# ---------------------------------------------------------------------------

@pytest.mark.real_data
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("real_data", _DENSE_TAGS, indirect=True)
def test_real_dense_linear_backward(real_data, use_sr_grad):
    """Inject the captured real upstream grad ``dy`` and compare dX/dW SNR.

    This is the most faithful real-data backward test: rather than an MSE
    against a synthetic target, we backprop the actual gradient that flowed
    through this layer during training.
    """
    assert real_data["kind"] == "dense_linear"
    x = real_data["x"].to(torch.bfloat16)
    w = real_data["w"].to(torch.bfloat16)
    dy = real_data["dy"].to(torch.bfloat16)

    # BF16 autograd reference.
    xr = x.clone().requires_grad_(True)
    wr = w.clone().requires_grad_(True)
    yr = torch.nn.functional.linear(xr, wr)
    yr.backward(dy)
    dx_ref, dw_ref = xr.grad.clone(), wr.grad.clone()

    def _grads(apply_fn):
        xq = x.clone().requires_grad_(True)
        wq = w.clone().requires_grad_(True)
        out = apply_fn(xq, wq)
        out.backward(dy)
        return xq.grad, wq.grad

    dx_nv, dw_nv = _grads(lambda xq, wq: NVFP4LinearFunction.apply(
        xq, wq, False, False, use_sr_grad, False))      # use_outer_scale=False
    dx_mx, dw_mx = _grads(lambda xq, wq: MXFP4LinearFunction.apply(
        xq, wq, False, False, use_sr_grad, False, "none", False, None))

    rows = []
    for name, ref, nv, mx in [("dX", dx_ref, dx_nv, dx_mx),
                              ("dW", dw_ref, dw_nv, dw_mx)]:
        nv_snr, mx_snr = calc_snr(ref, nv), calc_snr(ref, mx)
        rows.append([name, f"{nv_snr:.2f}", f"{mx_snr:.2f}",
                     f"{calc_cossim(ref, nv):.4f}", f"{calc_cossim(ref, mx):.4f}",
                     f"{nv_snr - mx_snr:+.2f}"])
        assert torch.isfinite(nv).all() and torch.isfinite(mx).all()
        assert nv_snr > _BWD_SNR_FLOOR, f"{name} NVFP4 SNR too low: {nv_snr:.2f}"
        assert mx_snr > _BWD_SNR_FLOOR, f"{name} MXFP4 SNR too low: {mx_snr:.2f}"

    _print_table(
        f"real dense bwd [{real_data['tag']}] K={x.shape[-1]} sr_grad={use_sr_grad}",
        rows,
    )


# ---------------------------------------------------------------------------
# Grouped GEMM (MoE mlp1) — forward
# ---------------------------------------------------------------------------

def _bf16_grouped_ref(x, w, num_tokens_per_expert):
    """Reference grouped matmul: per-expert ``x_e @ w_e.T`` with w as [E,N,K]."""
    outs = []
    start = 0
    for e in range(w.shape[0]):
        n = int(num_tokens_per_expert[e].item())
        if n == 0:
            continue
        xe = x[start:start + n].float()
        outs.append((xe @ w[e].float().T).to(x.dtype))
        start += n
    return torch.cat(outs, dim=0)


@pytest.mark.real_data
@pytest.mark.parametrize("real_data", _GROUPED_TAGS, indirect=True)
def test_real_grouped_gemm_forward(real_data):
    assert real_data["kind"] == "grouped_gemm"
    x = real_data["x"].to(torch.bfloat16)
    w = real_data["w"].to(torch.bfloat16)              # [E, N, K]
    ntpe = real_data["num_tokens_per_expert"].to(torch.int32)
    expert_indices = torch.repeat_interleave(
        torch.arange(w.shape[0], device=x.device, dtype=torch.int32), ntpe
    )

    y_ref = _bf16_grouped_ref(x, w, ntpe)
    y_nv = nvfp4_grouped_gemm(
        x, w, expert_indices, trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=False,
    )
    y_mx = mxfp4_grouped_gemm(
        x, w, expert_indices, trans_weights=True,
        use_2dblock_x=False, use_2dblock_w=False, use_sr_grad=False,
        clip_mode="none",
    )

    nv_snr, mx_snr = calc_snr(y_ref, y_nv), calc_snr(y_ref, y_mx)
    _print_table(
        f"real grouped fwd [{real_data['tag']}] K={x.shape[-1]} E={w.shape[0]}",
        [["O", f"{nv_snr:.2f}", f"{mx_snr:.2f}",
          f"{calc_cossim(y_ref, y_nv):.4f}", f"{calc_cossim(y_ref, y_mx):.4f}",
          f"{nv_snr - mx_snr:+.2f}"]],
    )

    assert torch.isfinite(y_nv).all() and torch.isfinite(y_mx).all()
    assert nv_snr > _FWD_SNR_FLOOR, f"NVFP4 grouped fwd SNR too low: {nv_snr:.2f}"
    assert mx_snr > _FWD_SNR_FLOOR, f"MXFP4 grouped fwd SNR too low: {mx_snr:.2f}"
