# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Testing helpers shared by ``mxfp4`` and ``nvfp4`` unit-test suites.

Until this module existed, each family kept its own ``tests/utils.py`` copy
of ``calc_snr`` / ``calc_cossim`` with slightly different return types and
edge-case handling.  Centralising them here:

* avoids the two copies drifting (one returned ``Tensor``, the other
  ``float``; one handled the ``noise == 0`` edge case, the other did not),
* gives one place to tighten numerical behaviour in the future,
* keeps the per-family ``tests/utils.py`` files free to hold only their
  format-specific helpers.

Both functions return Python ``float``; ``calc_snr`` safely returns
``+inf`` when the prediction matches the reference exactly.
"""

import statistics

import torch


def calc_snr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Signal-to-Noise Ratio (dB) of ``y`` against reference ``x``.

    ``SNR = 10 * log10(sum(x**2) / sum((x - y)**2))``

    Both inputs are cast to ``float32`` before accumulation so the result is
    stable even when the operands are BF16 / FP16 and the squared sums would
    overflow half-precision.  If the two tensors are bit-identical the SNR
    is ``+inf`` rather than ``NaN``.
    """
    signal = torch.sum(x.float() ** 2)
    noise = torch.sum((x.float() - y.float()) ** 2)
    if noise == 0:
        return float("inf")
    return (10 * torch.log10(signal / noise)).item()


def calc_cossim(x: torch.Tensor, y: torch.Tensor) -> float:
    """Cosine similarity between two tensors, computed in ``float32``.

    ``cossim = <x, y> / (||x|| * ||y||)``, flattened over all dims.
    """
    x_flat = x.reshape(-1).float()
    y_flat = y.reshape(-1).float()
    return (torch.dot(x_flat, y_flat) / (x_flat.norm() * y_flat.norm())).item()


def _nvfp4_autograd_snr_thresholds(
    *,
    K: int,
    use_sr_grad: bool,
    kind: str,
) -> tuple[float, float]:
    """Return ``(hard_floor, aggregate_floor)`` for NVFP4 autograd SNR tests.

    The floors are intentionally K-aware.  Large reduction dimensions amplify
    stochastic-rounding noise roughly as ``sqrt(K)``, so production-scale
    stress cases such as K=2048 need a lower per-tensor hard floor while small
    and medium cases should keep much tighter checks.
    """
    if kind == "nvfp4_linear":
        if K >= 1024:
            return (3.0, 4.5) if use_sr_grad else (4.0, 8.0)
        if K >= 256:
            return (10.0, 12.0) if use_sr_grad else (12.0, 14.0)
        return (12.0, 14.0) if use_sr_grad else (14.0, 15.0)

    if kind == "nvfp4_linear_outer_block":
        # Outer-block scaling on a per-128 grid contains every outlier within
        # a single FP32 outer scale, so the inner E4M3 scale always sits in
        # its sweet spot.  Empirically this is +2 dB tighter than the PTS
        # linear path; reflect that in the regression contract so a quiet
        # regression on the outer-block QDQ path (e.g. a clamp bug, a
        # mismatched axis) is caught even when the broader nvfp4_linear
        # thresholds would still pass.
        if K >= 1024:
            return (5.0, 6.5) if use_sr_grad else (6.0, 10.0)
        if K >= 256:
            return (12.0, 14.0) if use_sr_grad else (14.0, 16.0)
        return (14.0, 16.0) if use_sr_grad else (16.0, 17.0)

    if kind == "nvfp4_grouped_gemm":
        if K >= 1024:
            return (4.0, 6.0) if use_sr_grad else (7.0, 10.0)
        return (7.0, 10.0) if use_sr_grad else (10.0, 12.0)

    raise ValueError(f"Unknown NVFP4 autograd SNR threshold kind: {kind!r}")


def check_nvfp4_autograd_snr(
    snrs: dict[str, float],
    *,
    K: int,
    use_sr_grad: bool,
    kind: str,
    context: str = "",
) -> None:
    """Validate NVFP4 autograd SNR with per-tensor and aggregate checks.

    This helper is meant for tests that compare a full low-precision autograd
    path against a BF16/FP32 reference and report SNR for ``O``/``dX``/``dW``.

    It performs two checks:

    1. A K-aware hard floor for every tensor, catching real correctness bugs
       such as axis misalignment, bad scale handling, or dropped gradients.
    2. A K-aware aggregate floor on both median and mean SNR, preventing a very
       low global hard floor for large-K SR stress cases from weakening all
       other cases in the matrix.
    """
    hard_floor, aggregate_floor = _nvfp4_autograd_snr_thresholds(
        K=K,
        use_sr_grad=use_sr_grad,
        kind=kind,
    )
    ctx = f"{context}: " if context else ""

    for name, value in snrs.items():
        assert value > hard_floor, (
            f"{ctx}{name} SNR too low: {value:.2f} dB "
            f"< hard_floor={hard_floor:.2f} dB "
            f"(K={K}, use_sr_grad={use_sr_grad}, kind={kind}). "
            "This likely indicates a real regression in quantization, "
            "axis alignment, scale handling, or gradient propagation."
        )

    values = list(snrs.values())
    median_snr = statistics.median(values)
    mean_snr = sum(values) / len(values)
    assert median_snr > aggregate_floor, (
        f"{ctx}median SNR too low: {median_snr:.2f} dB "
        f"< aggregate_floor={aggregate_floor:.2f} dB "
        f"(values={snrs}, K={K}, use_sr_grad={use_sr_grad}, kind={kind})."
    )
    assert mean_snr > aggregate_floor, (
        f"{ctx}mean SNR too low: {mean_snr:.2f} dB "
        f"< aggregate_floor={aggregate_floor:.2f} dB "
        f"(values={snrs}, K={K}, use_sr_grad={use_sr_grad}, kind={kind})."
    )
