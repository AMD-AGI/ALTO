# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Per-slot Hadamard transform modes for the AdaHOP-in-ALTO port.

Mirrors AdaHOP's ``TransformMode`` Literal (see reference below) so JSON
configs produced by AdaHOP's calibration deserialize without translation.

Reference: 3rdparty/adahop/torchtitan/experiments/kernels/mxfp4/transform_config.py:21
"""

from typing import Dict, Literal, Tuple

TransformMode = Literal[
    "none",
    "hadamard",
    "outer_hadamard",
    "inner_outlier_extract_left",
    "inner_outlier_extract_left_col",
    "inner_outlier_extract_right",
    "outer_outlier_extract_left",
    "outer_outlier_extract_right",
    "full_precision",
]

OutlierPattern = Literal["row", "col", "none"]

VALID_MODES: frozenset = frozenset([
    "none",
    "hadamard",
    "outer_hadamard",
    "inner_outlier_extract_left",
    "inner_outlier_extract_left_col",
    "inner_outlier_extract_right",
    "outer_outlier_extract_left",
    "outer_outlier_extract_right",
    "full_precision",
])

# Modes implemented in the ALTO-side MXFP4LinearFunction port.
# Other VALID_MODES are accepted by the schema but will raise at runtime if
# encountered in a calibration JSON until their kernel paths are ported.
SUPPORTED_MODES: frozenset = frozenset([
    "none",
    "hadamard",
    "outer_hadamard",
    "full_precision",
    "inner_outlier_extract_left",
    "inner_outlier_extract_right",
])


def assert_mode_supported(mode: str, slot: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown TransformMode {mode!r} for slot {slot!r}")
    if mode not in SUPPORTED_MODES:
        raise NotImplementedError(f"TransformMode {mode!r} (slot {slot!r}) is not yet implemented in ALTO. "
                                  f"Supported: {sorted(SUPPORTED_MODES)}.")


PatternPair = Tuple[OutlierPattern, OutlierPattern]
PerSlotModes = Dict[Literal["forward_y", "backward_gx", "backward_gw"], TransformMode]
