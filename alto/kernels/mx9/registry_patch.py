# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Runtime patches that wire emulated formats into the standard quant path.

Importing this module injects a real ``format`` field into
``compressed_tensors.QuantizationArgs`` so recipe values like ``format: mx9``
survive pydantic parsing and become readable via ``getattr(args, "format", None)``
(by default unknown fields are silently dropped).

The actual ``fake_quantize`` dispatch (``args.format == "mx9"`` -> mx9 kernel)
lives in ``alto.models.patcher.ModelPatcher.patch_fake_quantize`` where the single
wrap of ``compressed_tensors...forward.fake_quantize`` already happens.
"""

from typing import Optional

_FORMAT_FIELD_INJECTED = False


def inject_format_field() -> None:
    """Add ``format: Optional[str] = None`` to ``QuantizationArgs`` (idempotent)."""
    global _FORMAT_FIELD_INJECTED
    if _FORMAT_FIELD_INJECTED:
        return

    from pydantic.fields import FieldInfo
    from compressed_tensors.quantization import QuantizationArgs, QuantizationConfig, QuantizationScheme

    if "format" not in QuantizationArgs.model_fields:
        QuantizationArgs.model_fields["format"] = FieldInfo(
            annotation=Optional[str], default=None
        )
        QuantizationArgs.model_rebuild(force=True)
        # QuantizationArgs is nested inside these models. Rebuild them as well so
        # recipe dictionaries with weights/input_activations.format are accepted
        # instead of being rejected by the old cached schema.
        QuantizationScheme.model_rebuild(force=True)
        QuantizationConfig.model_rebuild(force=True)

    _FORMAT_FIELD_INJECTED = True


inject_format_field()
