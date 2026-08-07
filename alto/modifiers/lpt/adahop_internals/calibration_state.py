# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Checkpointable AdaHOP calibration state.

Strategy-2 port (see /home/alirezak/han_branch/porting_notes.txt). Mirrors the
upstream adahop design
(3rdparty/adahop/torchtitan/experiments/kernels/mxfp4/transform_config.py:
``CalibrationStateManager`` + global state dict), adapted to ALTO.

The per-layer transform modes and the "calibration completed" flag live in a
module-level dict here -- NOT on any tensor subclass. ``CalibrationStateManager``
implements the PyTorch DCP ``Stateful`` interface so this state is saved and
restored alongside model/optimizer state. The bytes are produced with
``pickle.dumps`` (same approach upstream uses) so DCP treats the value as an
opaque non-tensor blob.

Lifecycle:
  * Fresh run: state starts empty (completed=False). The AdaHOPModifier observes
    patterns for ``calibration_steps`` and then calls :func:`set_calibration_result`
    which records the modes and flips completed=True.
  * Resume: CheckpointManager.load() calls ``load_state_dict`` which repopulates
    the module-level dict BEFORE the modifier inspects it on the first training
    step, so the modifier applies the stored modes and skips re-calibration.
"""

import pickle
from typing import Any, Dict

from torch.distributed.checkpoint.stateful import Stateful

__all__ = [
    "CalibrationStateManager",
    "get_adahop_calibration_state",
    "is_calibration_completed",
    "get_calibration_modes",
    "set_calibration_result",
    "reset_calibration_state",
]


# Module-level calibration state. Plain, fully-picklable Python objects only.
_STATE: Dict[str, Any] = {
    "completed": False,          # bool: has Phase B (mode selection) finished?
    "step_idx": 0,               # int: calibration steps observed so far
    "modes_by_fqn": {},          # {fqn: {forward_y, backward_gx, backward_gw}}
}


def get_calibration_state_dict() -> Dict[str, Any]:
    """Return the live module-level calibration state dict (mutable)."""
    return _STATE


def is_calibration_completed() -> bool:
    return bool(_STATE.get("completed", False))


def get_calibration_modes() -> Dict[str, Dict[str, str]]:
    return _STATE.get("modes_by_fqn", {})


def set_calibration_result(modes_by_fqn: Dict[str, Dict[str, str]], step_idx: int) -> None:
    """Record the aggregated per-layer modes and mark calibration complete."""
    _STATE["modes_by_fqn"] = dict(modes_by_fqn)
    _STATE["step_idx"] = int(step_idx)
    _STATE["completed"] = True


def set_step_idx(step_idx: int) -> None:
    _STATE["step_idx"] = int(step_idx)


def reset_calibration_state() -> None:
    """Reset to the fresh-run defaults (used by tests)."""
    _STATE["completed"] = False
    _STATE["step_idx"] = 0
    _STATE["modes_by_fqn"] = {}


class CalibrationStateManager(Stateful):
    """DCP ``Stateful`` wrapper around the module-level calibration state.

    Registered into ``CheckpointManager.states`` (from ``alto/train.py``) under
    the key ``"adahop_calibration"`` so calibration results round-trip with the
    checkpoint. The payload is pickled to keep DCP handling simple (opaque bytes,
    same pattern as the upstream reference).
    """

    _STATE_KEY = "adahop_calibration_data"

    def state_dict(self) -> Dict[str, Any]:
        return {self._STATE_KEY: pickle.dumps(_STATE)}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not state_dict or self._STATE_KEY not in state_dict:
            return
        loaded = pickle.loads(state_dict[self._STATE_KEY])
        if isinstance(loaded, dict):
            _STATE.update(loaded)


# Process-wide singleton so the modifier and the Trainer (alto/train.py) share
# one instance/state.
_singleton: CalibrationStateManager | None = None


def get_adahop_calibration_state() -> CalibrationStateManager:
    global _singleton
    if _singleton is None:
        _singleton = CalibrationStateManager()
    return _singleton
