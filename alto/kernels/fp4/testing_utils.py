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
