# Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from importlib.util import find_spec
from typing import Any, List

import torch
import torch.nn as nn

from torchtitan.components.quantization import (
    QuantizationConverter,
)
from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.models.moe.utils import set_token_group_alignment_size_m
from torchtitan.protocols.model_converter import register_model_converter
from torchtitan.tools.logging import logger

from modeloptimizer.config import QuantConfig


class ModelOptConverter(QuantizationConverter):

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        super().__init__(job_config, parallel_dims)

        self.quant_config: QuantConfig = job_config.quantization

        # TODO: set gmm alignment to quantization blocksize
        # set_token_group_alignment_size_m(ALIGN_SIZE_M)
        # logger.info(
        #     f"Setting token group alignment size to {ALIGN_SIZE_M}")


    def convert(self, model: nn.Module):
        print(self.quant_config)
        raise NotImplementedError

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]):
        return

    def post_initialization(self, model: nn.Module | list[nn.Module]):
        return


register_model_converter(ModelOptConverter, "modeloptimizer")
