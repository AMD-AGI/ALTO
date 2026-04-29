# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped GEMM autotune / shared constants.

Currently only hosts ``ALIGN_SIZE_M`` and ``QDQ_AXIS0_BLOCK_SIZE``. The file
name is kept for structural parity with
``alto/kernels/fp4/mxfp4/mxfp_grouped_gemm/autotune.py``; Triton autotune
configs will land here once a native NVFP4 grouped kernel is added.
"""

ALIGN_SIZE_M: int = 128
QDQ_AXIS0_BLOCK_SIZE: int = 16
