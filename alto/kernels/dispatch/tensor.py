# modified from https://github.com/pytorch/ao/blob/5ebd10d003b1fe6c9330f802ab9741ffde0bb7a9/torchao/prototype/moe_training/tensor.py
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch import nn
from torch._prims_common import suggest_memory_format
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchao.utils import TorchAOBaseTensor
from torchtitan.tools.logging import logger

from alto.kernels.fp4.mxfp4.mxfp_linear import _to_mxfp4_then_scaled_mm
from alto.kernels.fp4.mxfp4.mxfp_grouped_gemm.functional import _quantize_then_mxfp_scaled_grouped_mm
from alto.kernels.fp4.nvfp4.nvfp_linear import _to_nvfp4_then_scaled_mm
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm.functional import (
    _quantize_then_nvfp4_scaled_grouped_mm,
)
from alto.kernels.mxfp8.mxfp8_linear import _to_mxfp8_then_scaled_mm
from .config import TrainingOpConfig

aten = torch.ops.aten

_ops_to_preserve_subclass = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,  # for *.to(dtype)
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
}

gemm_ops = ("linear", "mm.default", "matmul.default", "addmm.default", "matmul")


class TrainingWeightWrapperBaseTensor(TorchAOBaseTensor):
    """
    A subclass of torch.Tensor that overrides the grouped_mm and linear ops
    to use dynamic quantization then low precision grouped_mm/linear op,
    based on the training config.
    """

    config: TrainingOpConfig | None = None

    @staticmethod
    def __new__(
        cls,
        tensor: torch.Tensor,
        config: TrainingOpConfig,
    ):
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.size(),
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            memory_format=suggest_memory_format(tensor),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=tensor.device,
            pin_memory=tensor.is_pinned(),
            requires_grad=tensor.requires_grad,
        )
        self.config = config
        return self

    def __init__(
        self,
        tensor: torch.Tensor,
        config: TrainingOpConfig,
    ):
        self._data = tensor
        self.config = config

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        raise NotImplementedError(
            f"{cls.__name__} not intended to be used directly, please override this method in a tensor subclass for your intended derived dtype."
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs={}):
        # unwrap args/kwargs and extract config
        config = None

        def unwrap(t):
            nonlocal config
            if config is None:
                config = t.config
            else:
                assert t.config == config, (
                    f"All TrainingWeightWrapperBaseTensor instances must have the same config, but found {t.config} and {config}"
                )
            return t._data

        args_unwrapped, kwargs_unwrapped = pytree.tree_map_only(TrainingWeightWrapperBaseTensor, unwrap,
                                                                (args, kwargs or {}))
        assert config is not None, (
            f"__torch_dispatch__ called on {func.__name__} without any TrainingWeightWrapperBaseTensor arguments")

        # detach is special case
        if func == torch.ops.aten.detach.default:
            return cls(args_unwrapped[0], config)

        # perform op
        out = func(*args_unwrapped, **kwargs_unwrapped)

        # return regular tensors for ops that don't preserve subclass
        if func not in _ops_to_preserve_subclass:
            return out

        # wrap outputs back into the same subclass for ops that do preserve subclass
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: cls(x, config),
            out,
        )

    def __repr__(self):
        return (f"TrainingWeightWrapperBaseTensor(data={self._data}, config={self.config})")

    def __tensor_flatten__(self):
        metadata = {
            "config": self.config,
        }
        return ["_data"], metadata

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, flatten_spec, outer_size, outer_stride):
        return cls(
            inner_tensors["_data"],
            flatten_spec["config"],
        )

    # fsdp hooks based on https://github.com/pytorch/pytorch/blob/20e40492b046b9287726d3ec656117e4dc38f0e2/test/distributed/_composable/fsdp/test_fully_shard_extensions.py#L81
    def fsdp_pre_all_gather(
        self,
        mesh: DeviceMesh,
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
        module: nn.Module,
        mp_policy: MixedPrecisionPolicy,
    ):
        # cast to mixed precision dtype prior to all-gather
        all_gather_inputs = (self._data.to(mp_policy.param_dtype),)
        all_gather_metadata = ()
        return all_gather_inputs, all_gather_metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[torch.Tensor] = None,
    ):
        (data,) = all_gather_outputs

        # For training step 1+, out=unsharded param.
        if out is not None:
            if isinstance(out, TrainingWeightWrapperBaseTensor):
                out_data = out._data
                out.config = self.config
            elif isinstance(out, DTensor) and isinstance(out._local_tensor, TrainingWeightWrapperBaseTensor):
                out_data = out._local_tensor._data
                out._local_tensor.config = self.config
            else:
                raise RuntimeError(
                    f"expect out to be TrainingWeightWrapperBaseTensor or DTensor with local_tensor=ScaledGroupedMM, but got {type(out)}"
                )

            # If `data` (all gather outputs) is already in the mixed precision policy param_dtype,
            # verify it has underlying storage as `out` (pre-allocated unsharded param),
            # and then we can just return directly.
            if data.dtype == param_dtype:
                assert (data.untyped_storage().data_ptr() == out_data.untyped_storage().data_ptr())
            else:
                # Otherwise, verify that `out` (pre-allocated unsharded param) has the
                # mixed precision policy param_dtype, then copy `data` to `out`.
                assert out_data.dtype == param_dtype, f"{out_data.dtype} {param_dtype}"
                out_data.copy_(data)

            return

        # For training step 0, out=None, so we need to return a new tensor of the same subclass.
        output = type(self)(data, self.config)
        inner_tensors = (data,)
        return output, inner_tensors


class MXFP4TrainingWeightWrapperTensor(TrainingWeightWrapperBaseTensor):
    """
    A subclass of torch.Tensor that overrides the grouped_mm and linear ops
    to use dynamic quantization of inputs to MXFP8, then runs the MXFP8 grouped_mm/linear,
    based on the training config.
    """

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # grouped_mm op override
        if func.__name__ == "_grouped_mm":
            # Use torchao scaled grouped mm with dynamic quant for
            # "2d x 3d with offsets" case (used for routed experts).
            # Otherwise, fall back to regular grouped mm.
            #
            # TODO: support "3d x 3d without offsets" case, which is
            # used for shared experts. This is basically the grouped_mm
            # kernel handling a bmm.
            A, B = args[0], args[1]
            bias = kwargs.get("bias", None)

            assert not isinstance(A, cls), f"A should not be a {cls.__name__}"

            assert isinstance(B, cls), f"B should be a {cls.__name__}"

            config = B.config
            A_is_2d = A.ndim == 2
            B_is_3d = B.ndim == 3
            offs = kwargs.get("offs", None)

            assert A_is_2d and B_is_3d and offs is not None, "Only 2d x 3d with offsets is supported for now"
            assert bias is None, "Bias is not supported for now"
            assert config.precision == "mxfp4", ("expected TrainingOpConfig with precision=mxfp4")

            # logger.info(
            #     f"[MXFP4GroupedMM]config: {config} A.shape: {A.shape} B.shape: {B.shape} offs.shape: {offs.shape}")

            return _quantize_then_mxfp_scaled_grouped_mm(
                A,
                B,
                offs=offs,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_dge=config.use_dge,
                use_hadamard=config.use_hadamard,
                clip_mode=config.clip_mode,
                use_macro_block_scaling=config.two_level_scaling == "blockwise",
            )

        # linear op override
        elif func.__name__ in gemm_ops:
            trans_b = func.__name__ == "linear"
            if func.__name__ == "addmm.default":
                bias, A, B = args[0], args[1], args[2]
            else:
                A, B = args[0], args[1]
                if len(args) > 2:
                    bias = args[2]
                else:
                    bias = None
            assert not isinstance(A, cls), f"A should not be a {cls.__name__} for func {func.__name__}"
            assert isinstance(B, cls), f"B should be a {cls.__name__} for func {func.__name__}"

            config = B.config
            assert config.precision == "mxfp4", ("expected TrainingOpConfig with precision=mxfp4")
            # logger.info(f"[MXFP4Linear]func: {func.__name__} config: {config}"
            #             f"A.shape: {A.shape} B.shape: {B.shape} bias.shape: {bias.shape if bias is not None else None}")

            Y = _to_mxfp4_then_scaled_mm(
                A,
                B if trans_b else B.T,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_dge=config.use_dge,
                clip_mode=config.clip_mode,
                use_hadamard=config.use_hadamard,
                use_macro_block_scaling=config.two_level_scaling == "blockwise",
            )
            if bias is not None:
                Y = Y + bias
            return Y
        else:
            # Disable torch_function by hand because we don't want
            # the wrapping behavior of the super() impl, go directly to dispatch
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


class NVFP4TrainingWeightWrapperTensor(TrainingWeightWrapperBaseTensor):
    """Weight tensor subclass that routes F.linear calls through NVFP4LinearFunction.

    The dispatch mechanism follows the same pattern as MXFP4TrainingWeightWrapperTensor:
    when PyTorch sees a linear/mm operation whose *weight* (B) is wrapped in this
    subclass, __torch_function__ is triggered and the NVFP4 QDQ path is used instead
    of the standard BF16 matmul.

    This means model code never needs to change — the training precision is
    controlled purely by wrapping the weight parameters via swap_params().

    Config fields used:
        use_2dblock_x  – 2D block scaling on activations
        use_2dblock_w  – 2D block scaling on weights (mirrors the axis-invariant view)
        use_sr_grad    – stochastic rounding on gradient quantization
        use_hadamard   – wgrad-path Hadamard rotation (mirrors MXFP4 behaviour)
        use_dge        – differentiable gradient estimator on wgrad (mirrors MXFP4)
        two_level_scaling – two-level NVFP4 scaling (global × per-block)

    Supported ops:
        linear / mm / addmm / matmul  – routed through ``_to_nvfp4_then_scaled_mm``
        _grouped_mm                    – routed through
                                         ``_quantize_then_nvfp4_scaled_grouped_mm``
                                         (2D × 3D with ``offs`` for MoE routed
                                         experts)

    """

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        # grouped_mm op override (MoE expert routing)
        if func.__name__ == "_grouped_mm":
            A, B = args[0], args[1]
            bias = kwargs.get("bias", None)
            offs = kwargs.get("offs", None)

            assert not isinstance(A, cls), f"A should not be a {cls.__name__}"
            assert isinstance(B, cls), f"B should be a {cls.__name__}"
            assert A.ndim == 2 and B.ndim == 3 and offs is not None, (
                "Only 2d x 3d with offsets is supported for NVFP4 grouped_mm"
            )
            assert bias is None, "Bias is not supported for grouped_mm"

            config = B.config
            assert config.precision == "nvfp4", (
                f"expected TrainingOpConfig with precision=nvfp4, got {config.precision}"
            )

            return _quantize_then_nvfp4_scaled_grouped_mm(
                A,
                B,
                offs=offs,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
            )

        # linear / mm overrides
        elif func.__name__ in gemm_ops:
            trans_b = func.__name__ == "linear"
            if func.__name__ == "addmm.default":
                bias, A, B = args[0], args[1], args[2]
            else:
                A, B = args[0], args[1]
                bias = args[2] if len(args) > 2 else None

            assert not isinstance(A, cls), (f"A should not be a {cls.__name__} for func {func.__name__}")
            assert isinstance(B, cls), (f"B should be a {cls.__name__} for func {func.__name__}")

            config = B.config
            assert config.precision == "nvfp4", (
                f"expected TrainingOpConfig with precision=nvfp4, got {config.precision}")

            # Pass the wrapper tensor itself into the autograd function —
            # matching the MXFP4 path — so that any upstream subclass
            # semantics (FSDP2 post-all-gather hooks, observability hooks
            # registered on __torch_dispatch__, subclass-preserving aten ops
            # in ``_ops_to_preserve_subclass``) reach the kernel boundary.
            # ``NVFP4LinearFunction.forward`` unwraps to ``._data`` at its
            # entry, so the autograd tape and downstream QDQ ops still see
            # plain tensors.
            W = B if trans_b else B.T
            Y = _to_nvfp4_then_scaled_mm(
                A,
                W,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
                use_sr_grad=config.use_sr_grad,
                use_outer_scale=config.two_level_scaling == "tensorwise",
                use_hadamard=config.use_hadamard,
                use_dge=config.use_dge,
                use_outer_block_scale=config.two_level_scaling == "outer_block",
                use_outer_2dblock_x=config.use_outer_2dblock_x,
                use_outer_2dblock_w=config.use_outer_2dblock_w,
                outer_block_size=config.outer_block_size,
            )
            if bias is not None:
                Y = Y + bias
            return Y
        else:
            # All other ops (copy_, view, ...) fall through without wrapping
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


class MXFP8TrainingWeightWrapperTensor(TrainingWeightWrapperBaseTensor):
    """Weight tensor subclass that routes linear calls through MXFP8LinearFunction."""

    _PRECISION_TO_FP8_VARIANT = {
        "mxfp8_e4m3": "e4m3",
        "mxfp8_e5m2": "e5m2",
    }

    @classmethod
    def __torch_function__(cls, func, types, args, kwargs={}):
        if func.__name__ == "_grouped_mm":
            raise NotImplementedError(
                "MXFP8 _grouped_mm is not supported by this dispatch path; "
                "restrict MXFP8 schemes to Linear targets."
            )

        if func.__name__ in gemm_ops:
            trans_b = func.__name__ == "linear"
            if func.__name__ == "addmm.default":
                bias, A, B = args[0], args[1], args[2]
            else:
                A, B = args[0], args[1]
                bias = args[2] if len(args) > 2 else None

            assert not isinstance(A, cls), (
                f"A should not be a {cls.__name__} for func {func.__name__}"
            )
            assert isinstance(B, cls), (
                f"B should be a {cls.__name__} for func {func.__name__}"
            )

            config = B.config
            assert config.precision in cls._PRECISION_TO_FP8_VARIANT, (
                "expected TrainingOpConfig with precision in "
                f"{tuple(cls._PRECISION_TO_FP8_VARIANT)}, got {config.precision}"
            )
            assert not config.use_hadamard and not config.use_dge, (
                "MXFP8 dispatch does not support Hadamard or DGE options."
            )

            Y = _to_mxfp8_then_scaled_mm(
                A,
                B if trans_b else B.T,
                fp8_variant=cls._PRECISION_TO_FP8_VARIANT[config.precision],
                use_sr_grad=config.use_sr_grad,
                use_2dblock_x=config.use_2dblock_x,
                use_2dblock_w=config.use_2dblock_w,
            )
            if bias is not None:
                Y = Y + bias
            return Y

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


torch.serialization.add_safe_globals([
    MXFP4TrainingWeightWrapperTensor,
    NVFP4TrainingWeightWrapperTensor,
    MXFP8TrainingWeightWrapperTensor,
])
