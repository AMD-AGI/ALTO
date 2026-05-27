# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Per-layer outlier-pattern aggregation and pattern-pair → TransformMode mapping.

Reference (AdaHOP, BSD-3):
  3rdparty/adahop/torchtitan/components/quantization/mx_calibration.py:158-265 (_aggregate_patterns)
  3rdparty/adahop/torchtitan/components/quantization/mx_calibration.py:412-481 (_apply_calibrated_transforms)

The aggregation logic is verbatim (Counter.most_common per tensor type). The
"T1..T8" pattern-pair construction is identical. The pair→mode lookup is also
identical; the only difference is that we return a dict of per-layer
(forward_y, backward_gx, backward_gw) modes instead of calling AdaHOP's global
``configure_layer_transforms``. The caller (the modifier) feeds the result
into the Phase-B wrapper constructors.
"""

from collections import Counter
from typing import Dict, List

from .transform_mode import OutlierPattern, PerSlotModes, TransformMode


def _opposite_pattern(pattern: OutlierPattern) -> OutlierPattern:
    if pattern == "row":
        return "col"
    if pattern == "col":
        return "row"
    return "none"


def aggregate_patterns(
    per_step_patterns: List[Dict[str, Dict[str, OutlierPattern]]],) -> Dict[str, Dict[str, OutlierPattern]]:
    """Reduce per-step pattern observations to one majority pattern per (layer, tensor).

    ``per_step_patterns[i][layer_name][tensor_name]`` is the pattern observed
    at step ``i`` for ``tensor_name`` in ``{"x", "w", "grad_output"}``. Missing
    entries are skipped; missing tensors default to ``"none"``.

    Returns ``{layer_name: {"x": ..., "w": ..., "grad_output": ...}}``.
    """
    all_layers = set()
    for step in per_step_patterns:
        all_layers.update(step.keys())

    final: Dict[str, Dict[str, OutlierPattern]] = {}
    for layer in sorted(all_layers):
        per_tensor: Dict[str, List[OutlierPattern]] = {"x": [], "w": [], "grad_output": []}
        for step in per_step_patterns:
            if layer not in step:
                continue
            for t in per_tensor:
                if t in step[layer]:
                    per_tensor[t].append(step[layer][t])

        final[layer] = {
            t: (Counter(per_tensor[t]).most_common(1)[0][0] if per_tensor[t] else "none") for t in per_tensor
        }
    return final


def patterns_to_modes(
    aggregated: Dict[str, Dict[str, OutlierPattern]],
    layer_transform_config: Dict[str, TransformMode],
) -> Dict[str, PerSlotModes]:
    """Map per-layer aggregated patterns to per-slot ``TransformMode`` triples.

    ``layer_transform_config`` is the recipe-supplied pattern-pair → mode lookup,
    e.g. ``{"row-row": "hadamard", "col-col": "full_precision", "none-none": "hadamard"}``.
    Unknown pairs fall back to ``"none"``.

    T1..T8 follow AdaHOP exactly:
      T1 = x_pat,  T2 = opposite(w_pat)            → forward_y      key "T1-T2"
      T4 = opposite(grad_output_pat), T5 = x_pat   → backward_gw    key "T4-T5"
      T7 = grad_output_pat, T8 = w_pat             → backward_gx    key "T7-T8"
    """
    out: Dict[str, PerSlotModes] = {}
    for layer, pats in aggregated.items():
        x_pat = pats.get("x", "none")
        w_pat = pats.get("w", "none")
        g_pat = pats.get("grad_output", "none")

        t1, t2 = x_pat, _opposite_pattern(w_pat)
        t4, t5 = _opposite_pattern(g_pat), x_pat
        t7, t8 = g_pat, w_pat

        out[layer] = {
            "forward_y": layer_transform_config.get(f"{t1}-{t2}", "none"),
            "backward_gw": layer_transform_config.get(f"{t4}-{t5}", "none"),
            "backward_gx": layer_transform_config.get(f"{t7}-{t8}", "none"),
        }
    return out
