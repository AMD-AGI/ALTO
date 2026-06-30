# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX packed-quantization Triton kernels (currently MX9 only).

Fake-quant reference / emulation lives in ``alto.modifiers.quantization.mx``.
This package provides GPU pack/unpack for MX block formats; it does not yet
include MX quantized GEMM or fused quant+matmul (see ``mxfp4`` / ``mxfp8``).
"""
