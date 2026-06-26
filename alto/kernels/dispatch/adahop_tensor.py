# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""AdaHOP-aware MXFP4 wrapper tensor subclass.

FSDP-execution fix: there is a SINGLE wrapper class.

* :class:`MXFP4AdaHOPWrapper` — used for ``scheme="mxfp4_adahop"`` from the
  moment the model is converted (so FSDP captures it as the param). It carries
  per-slot transform modes ``(forward_y_mode, backward_gx_mode,
  backward_gw_mode)`` and an optional ``HadamardTransform``. During calibration
  all three modes are ``"none"`` and the linear dispatch defers to plain MXFP4.
  When calibration finishes the modifier sets the chosen modes IN PLACE via
  :meth:`set_modes` (no new ``nn.Parameter``, no wrapper-type swap), so the
  modes take effect on the FSDP-owned tensor and the AdaHOP linear function
  actually runs in the forward.

Why single-wrapper + in-place modes (vs. the old two-wrapper swap): under
``fully_shard`` the param is captured at convert time. Replacing
``module.weight`` with a new Parameter at Phase B does NOT reach the tensor FSDP
all-gathers into ``F.linear`` (it keeps using the original), so AdaHOP never
executes. Keeping one wrapper and mutating its modes in place updates the
canonical FSDP-owned tensor; ``fsdp_post_all_gather`` propagates the modes to
the unsharded instance used in the forward.

Calibration observation uses transient *module* hooks (see
``calibration_hooks.py``); nothing unpicklable is stored on the tensor.
"""

from typing import Any, Optional, Tuple

import torch
from torchtitan.tools.logging import logger

from alto.modifiers.lpt.adahop_internals.mxfp4_linear_function import MXFP4AdaHOPLinearFunction
from alto.modifiers.lpt.adahop_internals.transform_mode import TransformMode, assert_mode_supported
from .tensor import MXFP4TrainingWeightWrapperTensor, gemm_ops


class MXFP4AdaHOPWrapper(MXFP4TrainingWeightWrapperTensor):
    """Single AdaHOP wrapper: per-slot modes (default ``"none"`` == plain MXFP4).

    During calibration all modes are ``"none"``; the modifier later flips them
    in place via :meth:`set_modes`.
    """

    @staticmethod
    def __new__(
        cls,
        tensor,
        config,
        *,
        hadamard_transform: Optional[Any] = None,
        forward_y_mode: TransformMode = "none",
        backward_gx_mode: TransformMode = "none",
        backward_gw_mode: TransformMode = "none",
    ):
        return super().__new__(cls, tensor, config)

    def __init__(
        self,
        tensor,
        config,
        *,
        hadamard_transform: Optional[Any] = None,
        forward_y_mode: TransformMode = "none",
        backward_gx_mode: TransformMode = "none",
        backward_gw_mode: TransformMode = "none",
    ):
        super().__init__(tensor, config)
        assert_mode_supported(forward_y_mode, "forward_y")
        assert_mode_supported(backward_gx_mode, "backward_gx")
        assert_mode_supported(backward_gw_mode, "backward_gw")
        self._hadamard_transform = hadamard_transform
        self._forward_y_mode = forward_y_mode
        self._backward_gx_mode = backward_gx_mode
        self._backward_gw_mode = backward_gw_mode

    def set_modes(
        self,
        forward_y_mode: TransformMode,
        backward_gx_mode: TransformMode,
        backward_gw_mode: TransformMode,
        hadamard_transform: Optional[Any] = None,
    ) -> None:
        """Set per-slot modes (and optionally the Hadamard transform) IN PLACE.

        Used at the Phase-A -> Phase-B transition, so no new ``nn.Parameter`` is
        created and the modes take effect on the FSDP-owned tensor.
        """
        assert_mode_supported(forward_y_mode, "forward_y")
        assert_mode_supported(backward_gx_mode, "backward_gx")
        assert_mode_supported(backward_gw_mode, "backward_gw")
        self._forward_y_mode = forward_y_mode
        self._backward_gx_mode = backward_gx_mode
        self._backward_gw_mode = backward_gw_mode
        if hadamard_transform is not None:
            self._hadamard_transform = hadamard_transform

    @property
    def is_calibrating(self) -> bool:
        """True while all slots are ``"none"`` (== plain MXFP4 / Phase A)."""
        return (
            self._forward_y_mode == "none"
            and self._backward_gx_mode == "none"
            and self._backward_gw_mode == "none"
        )

    def _adahop_kwargs(self) -> dict:
        return {
            "hadamard_transform": self._hadamard_transform,
            "forward_y_mode": self._forward_y_mode,
            "backward_gx_mode": self._backward_gx_mode,
            "backward_gw_mode": self._backward_gw_mode,
        }

    def __repr__(self):
        ht_state = "attached" if self._hadamard_transform is not None else "none"
        return (f"MXFP4AdaHOPWrapper(shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"forward_y={self._forward_y_mode}, backward_gx={self._backward_gx_mode}, "
                f"backward_gw={self._backward_gw_mode}, hadamard={ht_state})")

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        if func.__name__ in gemm_ops:
            x, weight, bias, trans_b = _extract_x_w_with_bias(func, args)
            assert isinstance(weight, cls), f"weight should be a {cls.__name__} for {func.__name__}"
            # During calibration (all slots "none") behave EXACTLY like plain
            # MXFP4: defer to the parent dispatch. Avoids the AdaHOP linear
            # function's "none" path (unused by the recipe post-Phase-B).
            if weight.is_calibrating:
                return super().__torch_function__(func, types, args, kwargs)
            operand_w = weight._data if trans_b else weight._data.T
            y = MXFP4AdaHOPLinearFunction.apply(
                x,
                operand_w,
                weight.config.use_sr_grad,
                weight._hadamard_transform,
                weight._forward_y_mode,
                weight._backward_gx_mode,
                weight._backward_gw_mode,
            )
            if bias is not None:
                y = y + bias
            return y
        # All other ops (grouped_mm, etc.) defer to the standard parent path.
        return super().__torch_function__(func, types, args, kwargs)

    def __tensor_flatten__(self):
        meta = {
            "config": self.config,
            "forward_y_mode": self._forward_y_mode,
            "backward_gx_mode": self._backward_gx_mode,
            "backward_gw_mode": self._backward_gw_mode,
        }
        return ["_data"], meta

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, flatten_spec, outer_size, outer_stride):
        instance = cls(
            inner_tensors["_data"],
            flatten_spec["config"],
            hadamard_transform=None,
            forward_y_mode=flatten_spec["forward_y_mode"],
            backward_gx_mode=flatten_spec["backward_gx_mode"],
            backward_gw_mode=flatten_spec["backward_gw_mode"],
        )
        return instance

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        # Step 1+: `out` is pre-allocated. Preserve our modes onto it.
        if out is not None:
            if isinstance(out, MXFP4AdaHOPWrapper):
                out._hadamard_transform = self._hadamard_transform
                out._forward_y_mode = self._forward_y_mode
                out._backward_gx_mode = self._backward_gx_mode
                out._backward_gw_mode = self._backward_gw_mode
            return super().fsdp_post_all_gather(all_gather_outputs, metadata, param_dtype, out=out)
        # Step 0: parent creates a fresh wrapper without modes; re-create with modes.
        (data,) = all_gather_outputs
        output = type(self)(data, self.config, **self._adahop_kwargs())
        return output, (data,)


def _extract_x_w_with_bias(func, args):
    """Return ``(x, weight, bias, trans_b)`` for a gemm-family op."""
    trans_b = func.__name__ == "linear"
    if func.__name__ == "addmm.default":
        bias, A, B = args[0], args[1], args[2]
    else:
        A, B = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
    return A, B, bias, trans_b


# Allowlist the wrapper for DCP checkpoint load (PyTorch >= 2.6 defaults
# torch.load to weights_only=True, which rejects unknown tensor-subclass globals).
# Mirrors the base-wrapper registration in dispatch/tensor.py.
torch.serialization.add_safe_globals([MXFP4AdaHOPWrapper])
