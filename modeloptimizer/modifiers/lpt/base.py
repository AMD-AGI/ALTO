from typing import Literal

import torch
from torch.nn import Module
from compressed_tensors.utils import match_named_modules
from pydantic import PrivateAttr
from torchtitan.models.common.moe.utils import set_token_group_alignment_size_m
from torchtitan.tools.logging import logger

from modeloptimizer.modifiers import Modifier
from modeloptimizer.kernels.dispatch import swap_params, MXFP4TrainingOpConfig
from modeloptimizer.kernels.mxfp4.mxfp_grouped_gemm.autotune import ALIGN_SIZE_M

__all__ = ["LowPrecisionTrainingModifier"]


class LowPrecisionTrainingModifier(Modifier):

    precision: Literal["mxfp4"] = "mxfp4"
    targets: list[str] = ["Linear"]
    ignore: list[str] = ["output"]

    use_2dblock_x: bool = False
    use_2dblock_w: bool = True
    use_hadamard: bool = False
    use_sr_grad: bool = False
    use_dge: bool = False

    _config: MXFP4TrainingOpConfig | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        set_token_group_alignment_size_m(ALIGN_SIZE_M)
        self._config = MXFP4TrainingOpConfig(
            use_2dblock_x=self.use_2dblock_x,
            use_2dblock_w=self.use_2dblock_w,
            use_hadamard=self.use_hadamard,
            use_sr_grad=self.use_sr_grad,
            use_dge=self.use_dge,
        )

    @property
    def requires_training_mode(self) -> bool:
        return True

    def on_convert(self, model: Module, **kwargs) -> bool:
        for name, module in match_named_modules(model, self.targets, self.ignore):
            if isinstance(module, torch.nn.Linear) or module.__class__.__name__.endswith("GroupedExperts"):
                swap_params(module, config=self._config)
            else:
                raise ValueError(f"Unsupported module type: {type(module)}")

        logger.info(f"LowPrecisionTrainingModifier converted model: {model}")
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
