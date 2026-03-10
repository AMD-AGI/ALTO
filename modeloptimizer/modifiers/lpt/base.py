from typing import Literal

import torch
from torch.nn import Module

from modeloptimizer.modifiers import Modifier
from modeloptimizer.modifiers.utils import HooksMixin

__all__ = ["LowPrecisionTraining"]


class LowPrecisionTraining(Modifier, HooksMixin):

    precision: Literal["mxfp4_1d2d"] = "mxfp4_1d2d"

    @property
    def requires_training_mode(self) -> bool:
        return True

    def on_initialize(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_pre_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_post_step(self, model_parts: list[Module], **kwargs) -> bool:
        return True

    def on_finalize(self, model_parts: list[Module], **kwargs) -> bool:
        return True
