# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX6 emulated (fake-quant) format.

MX6 shares the same Quark ``fake_quantize_mx6_mx9`` math as MX9 and only differs
in the element integer bit width (``quant_bit=5`` for MX6, ``8`` for MX9).

The ``QuantizationArgs.format`` field injection (needed so ``format: mx6``
recipes parse) lives in ``alto.kernels.mx9.registry_patch``; import it here so
this package is self-sufficient instead of relying on mx9 being imported first.
"""

from alto.kernels.mx9 import registry_patch  # noqa: F401  injects the QuantizationArgs.format field; do not remove
