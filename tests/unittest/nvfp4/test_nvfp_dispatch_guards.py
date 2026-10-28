# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""NVFP4 routing through ``lpt_linear`` and ``lpt_grouped_mm``."""

import pytest
import torch

import alto.nn.lpt as lpt
from alto.config import TrainingOpConfig
from alto.kernels.fp4.nvfp4.nvfp_grouped_gemm import ALIGN_SIZE_M
from alto.nn.lpt import lpt_module
from torchtitan.models.common.linear import Linear

LPTLinear = lpt_module(Linear)


def _make_config(**overrides) -> TrainingOpConfig:
    defaults = dict(
        precision="nvfp4",
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
        pytest.skip("NVFP4 routing tests require a CUDA device")
    return torch.device("cuda")


def test_grouped_mm_smoke(device):
    cfg = _make_config()
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
    assert output.abs().max().item() > 0


def test_grouped_mm_routes_to_nvfp4_grouped_kernel(monkeypatch, device):
    calls = []

    def _mock_grouped(A, B, *, offs, use_2dblock_x, use_2dblock_w,
                      use_sr_grad, use_outer_scale, use_hadamard, use_dge,
                      scale_format="e4m3"):
        calls.append({
            "A": A,
            "B": B,
            "offs": offs,
            "use_2dblock_x": use_2dblock_x,
            "use_2dblock_w": use_2dblock_w,
            "use_sr_grad": use_sr_grad,
            "use_outer_scale": use_outer_scale,
            "use_hadamard": use_hadamard,
            "use_dge": use_dge,
            "scale_format": scale_format,
        })
        return A.new_full((A.shape[0], B.shape[-1]), 7.0)

    monkeypatch.setattr(lpt, "_quantize_then_nvfp4_scaled_grouped_mm", _mock_grouped)

    cfg = _make_config(
        use_2dblock_x=True,
        use_2dblock_w=True,
        use_sr_grad=True,
        use_hadamard=True,
        use_dge=True,
        two_level_scaling="tensorwise",
    )
    num_experts, features_in, features_out = 2, 16, 32
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
    assert torch.all(output == 7)
    assert len(calls) == 1
    call = calls[0]
    assert call["A"] is activation
    assert call["offs"] is offsets
    assert call["B"].shape == (num_experts, features_in, features_out)
    assert call["use_2dblock_x"] is True
    assert call["use_2dblock_w"] is True
    assert call["use_sr_grad"] is True
    assert call["use_outer_scale"] is True
    assert call["use_hadamard"] is True
    assert call["use_dge"] is True
    assert call["scale_format"] == "e4m3"


@pytest.mark.parametrize("use_hadamard,use_dge", [
    (True, False),
    (False, True),
    (True, True),
])
def test_grouped_mm_recipe_surface_smoke(device, use_hadamard, use_dge):
    cfg = _make_config(
        use_2dblock_x=False,
        use_2dblock_w=True,
        use_sr_grad=True,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
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
    assert output.abs().max().item() > 0


@pytest.mark.parametrize("use_2dblock_x", [False, True])
@pytest.mark.parametrize("use_2dblock_w", [False, True])
@pytest.mark.parametrize("use_sr_grad", [False, True])
@pytest.mark.parametrize("use_hadamard", [False, True])
@pytest.mark.parametrize("use_dge", [False, True])
def test_linear_supported_knobs_still_work(
    device, use_2dblock_x, use_2dblock_w, use_sr_grad, use_hadamard, use_dge,
):
    cfg = _make_config(
        use_2dblock_x=use_2dblock_x,
        use_2dblock_w=use_2dblock_w,
        use_sr_grad=use_sr_grad,
        use_hadamard=use_hadamard,
        use_dge=use_dge,
    )
    activation = torch.randn(64, 32, dtype=torch.bfloat16, device=device)
    weight = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    output = lpt.lpt_linear(activation, weight, None, cfg)
    assert output.shape == (64, 32)
    assert output.dtype == torch.bfloat16


def test_linear_routes_to_nvfp4_linear_kernel(monkeypatch, device):
    calls = []

    def _mock_linear(A, B, *, use_2dblock_x, use_2dblock_w, use_sr_grad,
                     use_outer_scale, use_hadamard, use_dge,
                     scale_format="e4m3"):
        calls.append({
            "A": A,
            "B": B,
            "use_2dblock_x": use_2dblock_x,
            "use_2dblock_w": use_2dblock_w,
            "use_sr_grad": use_sr_grad,
            "use_outer_scale": use_outer_scale,
            "use_hadamard": use_hadamard,
            "use_dge": use_dge,
            "scale_format": scale_format,
        })
        return A.new_full((A.shape[0], B.shape[0]), 5.0)

    monkeypatch.setattr(lpt, "_to_nvfp4_then_scaled_mm", _mock_linear)

    cfg = _make_config(
        use_2dblock_x=True,
        use_2dblock_w=True,
        use_sr_grad=True,
        use_hadamard=True,
        use_dge=True,
        two_level_scaling="tensorwise",
    )
    activation = torch.randn(8, 16, dtype=torch.bfloat16, device=device)
    weight = torch.randn(32, 16, dtype=torch.bfloat16, device=device)
    output = lpt.lpt_linear(activation, weight, None, cfg)
    assert output.shape == (8, 32)
    assert torch.all(output == 5)
    assert len(calls) == 1
    call = calls[0]
    assert call["A"] is activation
    assert call["B"].shape == weight.shape
    assert call["use_2dblock_x"] is True
    assert call["use_2dblock_w"] is True
    assert call["use_sr_grad"] is True
    assert call["use_outer_scale"] is True
    assert call["use_hadamard"] is True
    assert call["use_dge"] is True
    assert call["scale_format"] == "e4m3"


def test_lpt_linear_populates_weight_grad(device):
    module = LPTLinear(
        LPTLinear.Config(
            in_features=32,
            out_features=32,
            bias=False,
            training_op_config=_make_config(),
        )
    ).to(device=device, dtype=torch.bfloat16)
    activation = torch.randn(16, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = module(activation)
    output.sum().backward()
    assert module.weight.grad is not None
    assert activation.grad is not None

    before = module.weight.detach().clone()
    torch.optim.AdamW(module.parameters(), lr=1e-1).step()
    difference = (module.weight.detach().float() - before.float()).norm().item()
    assert difference > 1e-6


@pytest.mark.parametrize("use_hadamard,use_dge", [
    (True, False),
    (False, True),
    (True, True),
])
def test_hadamard_and_dge_flow_grads(device, use_hadamard, use_dge):
    module = LPTLinear(
        LPTLinear.Config(
            in_features=32,
            out_features=32,
            bias=False,
            training_op_config=_make_config(use_hadamard=use_hadamard, use_dge=use_dge),
        )
    ).to(device=device, dtype=torch.bfloat16)
    activation = torch.randn(64, 32, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = module(activation)
    output.sum().backward()
    assert module.weight.grad is not None
    assert activation.grad is not None
