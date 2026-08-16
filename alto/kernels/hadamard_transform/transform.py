# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
from torch import Tensor, device
from torch.distributed.tensor import DTensor, Replicate

from .matrix import multihead_matmul
from .hadamard import deterministic_hadamard_matrix, random_hadamard_matrix


class HadamardFactory:
    """
    Factory used to apply hadamard transforms to a model.

    Class-level configuration parameters can be set using the configure() method.
    Individual create_transform() calls can override these defaults.

    Class attributes:
        block_size: Default size of the Hadamard block
        randomized: Default whether to use randomized Hadamard transform
        dtype: Default data type for the transform
        seed: Default random seed used for randomization
    """

    # Class-level default configuration
    block_size: int = 32
    randomized: bool = True
    dtype: torch.dtype = torch.float32
    transform_type: str = "default"
    seed: Optional[int] = None
    generator: torch.Generator = torch.Generator()
    _cached_transform: Optional['HadamardTransform'] = None

    @classmethod
    def configure(
        cls,
        block_size: Optional[int] = None,
        randomized: Optional[bool] = None,
        dtype: Optional[torch.dtype] = None,
        transform_type: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        Configure class-level default parameters for HadamardFactory.

        :param block_size: Default size of the Hadamard block
        :param randomized: Default whether to use randomized Hadamard transform
        :param dtype: Default data type for the transform
        :param seed: Default random seed used for randomization
        """
        if block_size is not None:
            cls.block_size = block_size
        if randomized is not None:
            cls.randomized = randomized
        if dtype is not None:
            cls.dtype = dtype
        if transform_type is not None:
            cls.transform_type = transform_type
        if seed is not None:
            cls.seed = seed
            cls.generator.manual_seed(seed)

    @classmethod
    def refresh(cls) -> None:
        """Clear the cached transform so the next create_transform generates a fresh one."""
        cls._cached_transform = None

    @classmethod
    def create_transform(
        cls,
        device: torch.device,
    ) -> 'HadamardTransform':
        """
        Create a HadamardTransform for applying to a module.

        Parameters default to class-level configuration set via configure().
        Any parameter explicitly provided will override the class default.

        :param device: Device to create the transform on
        :return: HadamardTransform instance
        """
        if cls._cached_transform is not None:
            return cls._cached_transform

        if cls.transform_type == "default":
            weight = cls._create_weight(device)
            perm = cls._create_permutation(weight) if cls.randomized else None
            t = HadamardTransform(weight, perm)
        elif cls.transform_type == "3rht":
            n = cls.block_size
            w = cls._create_weight(device)
            p = cls._create_permutation(w)
            combined = w[p][:, p]

            w = cls._create_weight(device)
            p = cls._create_permutation(w)
            combined = combined @ (w[p][:, p])

            w = cls._create_weight(device)
            p = cls._create_permutation(w)
            combined = combined @ (w[p][:, p])

            combined = combined / n
            t = HadamardTransform(combined, perm=None)
        else:
            raise NotImplementedError("transform_type options are: default and 3rht")

        cls._cached_transform = t
        return t

    @classmethod
    def _create_weight(
        cls,
        device: device,
    ) -> Tensor:
        if not cls.randomized:
            data = deterministic_hadamard_matrix(cls.block_size, cls.dtype, device)
        else:
            data = random_hadamard_matrix(cls.block_size, cls.dtype, device, cls.generator)
        return data

    @classmethod
    def _create_permutation(cls, weight: Tensor) -> Tensor:
        data = torch.randperm(weight.size(0), generator=cls.generator)
        return data


class HadamardTransform:
    """
    Hadamard transform that can be applied to tensors.

    :param weight: Hadamard matrix
    :param perm: Optional permutation tensor for randomized transforms
    """

    def __init__(
        self,
        weight: Tensor,
        perm: Optional[Tensor],
    ):
        scale = weight.size(0) ** 0.5
        if perm is not None:
            weight = weight[perm][:, perm]
        if isinstance(weight, DTensor):
            assert weight.placements[0] == Replicate()
            weight = weight.to_local()
        self.weight = (weight / scale).contiguous()

    def __call__(self, value: Tensor, inverse: bool = False, left_mul: bool = False) -> Tensor:
        """
        Apply the Hadamard transform to a tensor.

        :param value: Input tensor to transform
        :param inverse: If True, apply the inverse transform
        :param left_mul: If True, multiply from the left (weight @ value),
                         else from the right (value @ weight)
        :return: Transformed tensor
        """
        weight = self.weight

        if inverse:
            weight = weight.T

        w = weight.to(device=value.device, dtype=value.dtype)
        if left_mul:
            return multihead_matmul(w, value)
        else:
            return multihead_matmul(value, w)
