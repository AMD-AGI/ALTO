# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX9 emulated (fake-quant) format.

NOTE: this package currently provides fake-quantization (QDQ emulation) only --
there is no real packed kernel / GEMM. It plugs into the standard
``QuantizationModifier`` path via runtime patches, without touching the generic
calibration / observer machinery. The patches that wire it into the quant path
(``QuantizationArgs.format`` field injection + ``fake_quantize`` mx9 dispatch)
live in ``registry_patch`` and ``alto.models.patcher``.
"""

from . import registry_patch  # noqa: F401  injects the QuantizationArgs.format field; do not remove
