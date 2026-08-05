# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Inject the QuantizationArgs.format field BEFORE importing QuantizationModifier:
# the modifier compiles its nested QuantizationScheme schema at class-definition
# time, so the field must exist first or recipes carrying ``format: mx6/mx9`` are
# rejected by the cached (format-less) schema.
from .format_registry import inject_format_field

inject_format_field()

from .base import QuantizationModifier
from .gptq import GPTQModifier
from .awq import AWQModifier

__all__ = [
    "QuantizationModifier",
    "GPTQModifier",
    "AWQModifier",
]
