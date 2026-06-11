# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

# Inject the QuantizationArgs.format field BEFORE importing modifiers: the alto
# QuantizationModifier compiles its nested QuantizationScheme schema at class
# definition time, so the field must exist first or recipes carrying
# ``format: mx6/mx9`` are rejected by the cached (format-less) schema. Calling
# inject_format_field() explicitly (it is idempotent) makes this ordering
# dependency self-documenting instead of relying on import order, which an
# import-sorter could silently reshuffle.
from .kernels.mx9.registry_patch import inject_format_field

inject_format_field()

from .components import *
from .config import *
from .observers import *
from .modifiers import *
from .models import *

# MX fake-quant kernels: import for their dispatch wiring. The format-field
# injection they depend on already ran above.
from .kernels import mx6  # noqa: F401,E402
from .kernels import mx9  # noqa: F401,E402
