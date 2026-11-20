# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Add ALTO's format, block_axis, and scale_type fields to QuantizationArgs.

Rebuild the nested pydantic schemas before recipes are parsed. MX dispatch and
INT8 scale calculation are installed by alto.patches.patcher.ModelPatcher.
"""

from typing import Literal, Optional

_FORMAT_FIELD_INJECTED = False

DEFAULT_BLOCK_AXIS = -1


def inject_format_field() -> None:
    """Add the ALTO fields to ``QuantizationArgs`` (idempotent)."""
    global _FORMAT_FIELD_INJECTED
    if _FORMAT_FIELD_INJECTED:
        return

    from pydantic.fields import FieldInfo
    from compressed_tensors.quantization import QuantizationArgs, QuantizationConfig, QuantizationScheme

    added = False
    for name, field in (
        ("format", FieldInfo(annotation=Optional[str], default=None)),
        ("block_axis", FieldInfo(annotation=int, default=DEFAULT_BLOCK_AXIS)),
        ("scale_type", FieldInfo(annotation=Literal["float", "pot"], default="float")),
    ):
        if name not in QuantizationArgs.model_fields:
            QuantizationArgs.model_fields[name] = field
            added = True

    if added:
        QuantizationArgs.model_rebuild(force=True)
        # QuantizationArgs is nested inside these models. Rebuild them as well so
        # recipe dictionaries with weights/input_activations.format are accepted
        # instead of being rejected by the old cached schema.
        QuantizationScheme.model_rebuild(force=True)
        QuantizationConfig.model_rebuild(force=True)

    _FORMAT_FIELD_INJECTED = True


def validate_scale_type(args):
    """PoT changes symmetric INT8 scales; tensor layout and QDQ stay upstream."""
    if args is None or args.scale_type == "float":
        return
    if not (args.num_bits == 8 and args.type == "int" and args.symmetric and args.format is None and
            args.scale_dtype is None):
        raise ValueError("scale_type=pot requires symmetric INT8 (num_bits=8, type=int), "
                         "with no format or scale_dtype override")


inject_format_field()
