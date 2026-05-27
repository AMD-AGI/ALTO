# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""AdaHOP-aware MXFP4 wrapper tensor subclasses.

Two child classes of ``MXFP4TrainingWeightWrapperTensor``:

* :class:`MXFP4CalibrationWrapper` — Phase A. Dispatch identical to its parent
  (plain MXFP4), plus an optional ``_calibration_callback`` invoked from
  ``__torch_function__`` before the standard path. Used during the 30-step
  outlier-pattern observation window.

* :class:`MXFP4AdaHOPWrapper` — Phase B. Carries frozen ``(forward_y_mode,
  backward_gx_mode, backward_gw_mode)`` and a ``HadamardTransform``; routes
  the ``linear`` dispatch through :class:`MXFP4AdaHOPLinearFunction` with
  those modes as ``apply()`` arguments. No global state, no FQN lookup.

Design notes (see ADAHOP_TO_ALTO_INTEGRATION_PLAN.md):

* Modes ride on the wrapper instance — the canonical instance that lives on
  ``nn.Parameter.data`` is the one PyTorch passes to ``__torch_function__``
  for the linear op, so the modes are reachable at the right moment.
* ``__torch_dispatch__`` is **not** overridden. Detach/view/copy re-wraps
  drop the modes; that is fine because those transient re-wraps never
  re-enter ``__torch_function__`` for ``linear`` — by then the canonical
  wrapper has already supplied modes to ``MXFP4AdaHOPLinearFunction``.
* ``__tensor_flatten__`` / ``__tensor_unflatten__`` round-trip the *string*
  modes so DCP checkpoints survive. The ``HadamardTransform`` is not
  serializable; on load it's restored to ``None`` and the modifier
  re-attaches a fresh one (or keeps ``None`` if the recipe is hadamard-free
  and only ``none``/``full_precision`` modes are used).
* ``fsdp_post_all_gather`` is overridden minimally to preserve modes on the
  fresh wrapper instance produced at training step 0.
"""

from typing import Any, Callable, Optional, Tuple

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchtitan.tools.logging import logger

from alto.modifiers.lpt.adahop_internals.mxfp4_linear_function import MXFP4AdaHOPLinearFunction
from alto.modifiers.lpt.adahop_internals.transform_mode import TransformMode, assert_mode_supported
from .tensor import MXFP4TrainingWeightWrapperTensor, TrainingWeightWrapperBaseTensor, gemm_ops

CalibrationCallback = Callable[[torch.Tensor, torch.Tensor], None]


class MXFP4CalibrationWrapper(MXFP4TrainingWeightWrapperTensor):
    """Phase-A wrapper: standard MXFP4 dispatch plus an optional observation hook."""

    @staticmethod
    def __new__(cls, tensor, config, *, calibration_callback: Optional[CalibrationCallback] = None):
        return super().__new__(cls, tensor, config)

    def __init__(self, tensor, config, *, calibration_callback: Optional[CalibrationCallback] = None):
        super().__init__(tensor, config)
        self._calibration_callback = calibration_callback

    def attach_calibration_callback(self, cb: Optional[CalibrationCallback]) -> None:
        self._calibration_callback = cb

    def __repr__(self):
        cb_state = "attached" if self._calibration_callback is not None else "none"
        return (f"MXFP4CalibrationWrapper(shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"callback={cb_state})")

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        if func.__name__ in gemm_ops:
            x, weight = _extract_x_w(func, args)
            if isinstance(weight, cls) and weight._calibration_callback is not None:
                # Detach to avoid building a graph on the observation path.
                weight._calibration_callback(x.detach(), weight._data.detach())
        return super().__torch_function__(func, types, args, kwargs)


class MXFP4AdaHOPWrapper(MXFP4TrainingWeightWrapperTensor):
    """Phase-B wrapper: frozen per-slot modes baked in at construction."""

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
        logger.info(f"[AdaHOP] Restored MXFP4AdaHOPWrapper from checkpoint: "
                    f"forward_y={instance._forward_y_mode}, "
                    f"backward_gx={instance._backward_gx_mode}, "
                    f"backward_gw={instance._backward_gw_mode}, "
                    f"hadamard_transform=None (will be re-bound by modifier on first use)")
        return instance

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        # Step 1+: `out` is pre-allocated. Preserve our modes onto it if it's a sibling instance.
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


def _extract_x_w(func, args):
    """Return ``(x, weight)`` for a gemm-family op. Mirrors parent dispatch."""
    if func.__name__ == "addmm.default":
        _, A, B = args[0], args[1], args[2]
    else:
        A, B = args[0], args[1]
    return A, B


def _extract_x_w_with_bias(func, args):
    """Return ``(x, weight, bias, trans_b)`` for a gemm-family op."""
    trans_b = func.__name__ == "linear"
    if func.__name__ == "addmm.default":
        bias, A, B = args[0], args[1], args[2]
    else:
        A, B = args[0], args[1]
        bias = args[2] if len(args) > 2 else None
    return A, B, bias, trans_b
