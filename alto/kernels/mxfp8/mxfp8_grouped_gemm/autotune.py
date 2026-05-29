# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Autotune configs for mxfp8 grouped GEMM.

v1 keeps a single conservative config:
- BLOCK_SIZE_K == QUANT_BLOCK_SIZE (=32) so each tl.dot_scaled call covers
  exactly one mx scale group; this matches the numerical contract validated
  by alto/kernels/mxfp8/mxfp8_linear.py.
- BSM=BSN=128 matches mxfp4 grouped GEMM's default tile.
Wider autotune is deferred to v2.
"""

import triton

ALIGN_SIZE_M = 128  # token routing alignment; tokens routed to the same expert must form contiguous blocks of this size

STANDARD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 32,
        },
        num_stages=2,
        num_warps=4,
    ),
]
