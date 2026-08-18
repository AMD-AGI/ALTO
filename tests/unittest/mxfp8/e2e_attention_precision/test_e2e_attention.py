# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Toy attention training sanity for low-precision attention kernels.

The grouped-GEMM E2E test trains a tiny MoE because that kernel's native data
shape is expert-routed matmul. Attention has different data: Q/K/V over a
sequence. This test keeps the same E2E idea, but uses a teacher-student toy
attention task so the signal belongs to the attention kernel, not MoE routing.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is required.")

from .e2e_attention_common import (
    AttentionExperimentConfig,
    run_forward_only_experiment,
    run_training_experiment,
    window_mean,
)


def _assert_finite_and_descending(result):
    _, bf16_tail = window_mean(result["paths"]["bf16 baseline"]["mean"])
    for name, data in result["paths"].items():
        losses = data["mean"]
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            f"{name} loss went non-finite: {losses}"
        head, tail = window_mean(losses)
        assert tail < head * 0.98, f"{name} loss did not trend down: head={head:.6f} tail={tail:.6f}"
        assert tail < bf16_tail * 2.0, \
            f"{name} final loss diverged from bf16: {name}_tail={tail:.6f} bf16_tail={bf16_tail:.6f}"


@pytest.mark.parametrize("causal", [True])
def test_toy_attention_trains_and_tracks_bf16(causal):
    device = torch.device("cuda")
    config = AttentionExperimentConfig(steps=40, causal=causal)
    result = run_training_experiment(config, seeds=(1234,), device=device)
    _assert_finite_and_descending(result)


@pytest.mark.parametrize("causal", [True])
def test_forward_only_tracks_bf16_trajectory(causal):
    device = torch.device("cuda")
    config = AttentionExperimentConfig(steps=40, causal=causal)
    result = run_forward_only_experiment(config, seeds=(1234,), device=device)
    _assert_finite_and_descending(result)
