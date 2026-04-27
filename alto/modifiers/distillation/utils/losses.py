# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

__all__ = ["LogitsDistillationLoss"]


class LogitsDistillationLoss(_Loss):
    """
    KL-divergence distillation loss over logits.

    This implements the standard knowledge distillation objective using:
      - student log-probabilities from softened student logits
      - teacher probabilities from softened teacher logits

    The result is multiplied by temperature^2, matching the common
    distillation scaling convention.
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "batchmean"):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if reduction not in {"none", "batchmean", "sum", "mean"}:
            raise ValueError("reduction must be one of: none, batchmean, sum, mean")

        self.temperature = float(temperature)
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student_logits and teacher_logits must have the same shape, "
                f"got {student_logits.shape} and {teacher_logits.shape}"
            )

        t = self.temperature

        student_log_probs = F.log_softmax(student_logits / t, dim=-1)
        teacher_probs = F.softmax(teacher_logits / t, dim=-1)

        num_classes = student_log_probs.size(-1)
        student_log_probs = student_log_probs.reshape(-1, num_classes)
        teacher_probs = teacher_probs.reshape(-1, num_classes)

        loss = F.kl_div(
            student_log_probs,
            teacher_probs.detach(),
            reduction=self.reduction,
        )

        return loss * (t * t)
