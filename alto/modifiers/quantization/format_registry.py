# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Runtime patch that wires emulated formats into the standard quant path.

Importing this module injects real ``format`` and ``block_axis`` fields into
``compressed_tensors.QuantizationArgs`` so recipe values like ``format: mx9``
survive pydantic parsing and become readable via
``getattr(args, "format", None)`` (by default unknown fields are silently
dropped).

``block_axis`` is the tensor axis the packed formats group elements along, and
defaults to the last one. A recipe only needs to set it where that is the wrong
axis, such as MX on Conv2d weights, whose blocks run along the input channels
rather than along the kernel width.

The actual ``fake_quantize`` dispatch (``args.format == "mx9"`` -> mx9) lives in
``alto.models.patcher.ModelPatcher.patch_fake_quantize`` where the single wrap of
``compressed_tensors...forward.fake_quantize`` already happens.

``inject_format_field()`` is called at the top of this package's ``__init__`` (before
``QuantizationModifier`` is imported) so the fields exist before the modifier
compiles its nested ``QuantizationScheme`` schema.
"""

from typing import Optional

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


inject_format_field()
