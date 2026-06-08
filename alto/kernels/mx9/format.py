# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""MX9 format constants (fake-quant emulation only; no real packed kernel)."""

# Matches Quark mx9 reference (test_mx.py uses block_size=16; Quark leaves it
# configurable via MX9Spec.block_size). 16 is the L2 block; the shared prime bit
# is shared across every SHARED_PRIME_BIT_GROUP (=2) elements.
BLOCK_SIZE = 16
QUANT_BIT = 8
SHARED_PRIME_BIT_GROUP = 2
