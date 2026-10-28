# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""AMD-FP4 routing through ``lpt_linear`` and ``lpt_grouped_mm``."""

from __future__ import annotations

import pytest
import torch

import alto.nn.lpt as lpt
from alto.kernels.dispatch.config import TrainingOpConfig
from alto.kernels.fp4.amdfp4 import ALIGN_SIZE_M


def _make_amdfp4_config(**overrides) -> TrainingOpConfig:
    defaults = dict(
        precision="amdfp4",
        use_2dblock_x=False,
        use_2dblock_w=False,
        use_hadamard=False,
        use_sr_grad=False,
        use_dge=False,
    )
    defaults.update(overrides)
    return TrainingOpConfig(**defaults)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("AMD-FP4 routing tests require a CUDA device")
    return torch.device("cuda")


def test_amdfp4_config_has_no_inner_scale_format_knob():
    cfg = _make_amdfp4_config()
    assert cfg.precision == "amdfp4"
    assert not hasattr(cfg, "inner_scale_format")


def test_amdfp4_linear_routes_to_amdfp4_kernel(monkeypatch, device):
    calls = []

    def _mock_amdfp4(A, W, *, use_2dblock_x, use_2dblock_w,
                     use_sr_grad, use_outer_scale, use_hadamard, use_dge):
        calls.append(use_outer_scale)
        return A.new_zeros((A.shape[0], W.shape[0]))

    def _forbidden_nvfp4(*args, **kwargs):
        raise AssertionError("precision='amdfp4' must not call the NVFP4 linear kernel")

    monkeypatch.setattr(lpt, "_to_amdfp4_then_scaled_mm", _mock_amdfp4)
    monkeypatch.setattr(lpt, "_to_nvfp4_then_scaled_mm", _forbidden_nvfp4)

    cfg = _make_amdfp4_config()
    activation = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
    weight = torch.randn(16, 16, dtype=torch.bfloat16, device=device)
    output = lpt.lpt_linear(activation, weight, None, cfg)
    assert output.shape == (16, 16)
    assert calls == [False]


def test_amdfp4_grouped_mm_routes_to_amdfp4_kernel(monkeypatch, device):
    calls = []

    def _mock_amdfp4_grouped(A, B, *, offs, use_2dblock_x, use_2dblock_w,
                             use_sr_grad, use_outer_scale, use_hadamard, use_dge):
        calls.append(B.shape)
        return torch.zeros(A.shape[0], B.shape[-1], dtype=A.dtype, device=A.device)

    def _forbidden_nvfp4_grouped(*args, **kwargs):
        raise AssertionError(
            "precision='amdfp4' grouped GEMM must not call the NVFP4 grouped kernel"
        )

    monkeypatch.setattr(lpt, "_quantize_then_amdfp4_scaled_grouped_mm", _mock_amdfp4_grouped)
    monkeypatch.setattr(lpt, "_quantize_then_nvfp4_scaled_grouped_mm", _forbidden_nvfp4_grouped)

    cfg = _make_amdfp4_config()
    num_experts, features_in, features_out = 2, 16, 16
    tokens = ALIGN_SIZE_M
    activation = torch.randn(tokens, features_in, dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        num_experts, features_out, features_in, dtype=torch.bfloat16, device=device
    )
    offsets = torch.tensor(
        [tokens // num_experts, tokens], dtype=torch.int32, device=device
    )
    lpt.lpt_grouped_mm(activation, weight, offsets, cfg)
    assert calls == [(num_experts, features_in, features_out)]


def test_amdfp4_linear_smoke(device):
    cfg = _make_amdfp4_config()
    activation = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    weight = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    output = lpt.lpt_linear(activation, weight, None, cfg)
    assert output.shape == (32, 32)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert output.abs().max().item() > 0


def test_amdfp4_grouped_mm_smoke(device):
    cfg = _make_amdfp4_config()
    num_experts, features_in, features_out = 2, 16, 16
    tokens = ALIGN_SIZE_M
    activation = torch.randn(tokens, features_in, dtype=torch.bfloat16, device=device)
    weight = torch.randn(
        num_experts, features_out, features_in, dtype=torch.bfloat16, device=device
    )
    offsets = torch.tensor(
        [tokens // num_experts, tokens], dtype=torch.int32, device=device
    )
    output = lpt.lpt_grouped_mm(activation, weight, offsets, cfg)
    assert output.shape == (tokens, features_out)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    assert output.abs().max().item() > 0
