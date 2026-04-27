# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch


def unwrap_weight_wrapper(weight: torch.Tensor) -> torch.Tensor:
    """Normalize wrapper-tensor inputs to a plain local ``torch.Tensor``.

    Both MXFP4 and NVFP4 dispatch paths may pass a tensor subclass as the
    ``weight`` argument into their autograd function boundary
    (e.g. ``MXFP4TrainingWeightWrapperTensor`` /
    ``NVFP4TrainingWeightWrapperTensor``).  At that point we want to save and
    manipulate the plain local tensor payload.

    Notes:

    * DTensor inputs are normally already unwrapped to their local tensors by
      PyTorch's DTensor dispatcher before these autograd functions run.
    * For wrapper tensors that explicitly carry a ``._data`` payload, prefer
      that payload.
    * For any other tensor subclass, ``as_subclass(torch.Tensor)`` strips the
      subclass type while preserving the underlying local tensor storage.
    """
    if type(weight) is torch.Tensor:
        return weight
    if hasattr(weight, "_data"):
        return weight._data  # type: ignore[return-value]
    return weight.as_subclass(torch.Tensor)
