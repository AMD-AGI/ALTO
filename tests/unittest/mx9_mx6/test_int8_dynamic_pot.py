# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""INT8 PoT API, runtime, and optional comparisons with the pylight reference.

Put a pylight checkout on PYTHONPATH to run the reference comparisons as well.
The fixed-vector and lifecycle tests do not depend on pylight.
"""

import copy
import importlib

import pytest
import torch
from torch import nn

from .test_quantization_api import apis

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU"))
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _pylight_recipe(apis, **kwargs):
    return apis.create_quantization_recipe(
        "int8_dynamic",
        scale_type="pot",
        activation_strategy="tensor",
        dynamic_weights=True,
        **kwargs,
    )


def test_scale_policy_preserves_other_dynamic_int8_settings(apis):
    recipe = apis.create_quantization_recipe("int8_dynamic", scale_type="pot")
    group = recipe.modifiers[0].resolved_config.config_groups["group_linear"]
    assert not group.weights.dynamic
    assert group.input_activations.dynamic
    assert group.input_activations.strategy == "token"
    assert group.weights.format is group.input_activations.format is None
    assert group.weights.scale_type == group.input_activations.scale_type == "pot"
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(3))
    apis.apply_quantization(model, recipe)
    x = torch.tensor([[[127.1, 2.5, 3.5], [1.271, 0.025, 0.035]]])
    expected = torch.tensor([[[128., 2., 4.], [1.265625, 0.03125, 0.03125]]])
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_scale_policy_reuses_upstream_range_reduction_and_qdq(apis, monkeypatch):
    forward = importlib.import_module("compressed_tensors.quantization.lifecycle.forward")
    helpers = importlib.import_module("compressed_tensors.quantization.utils.helpers")
    forward_helpers = importlib.import_module("compressed_tensors.quantization.lifecycle.forward_helpers")
    assert forward.compute_dynamic_scales_and_zp is helpers.compute_dynamic_scales_and_zp
    original = forward_helpers._quantize_dequantize
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["args"].scale_type)
        return original(*args, **kwargs)

    monkeypatch.setattr(forward_helpers, "_quantize_dequantize", counted)
    for scale_type in ("float", "pot"):
        model = nn.Sequential(nn.Linear(3, 3))
        recipe = apis.create_quantization_recipe("int8_dynamic", scale_type=scale_type, dynamic_weights=True)
        apis.apply_quantization(model, recipe)
        assert torch.isfinite(model(torch.randn(1, 2, 3))).all()
    assert calls == ["float", "float", "pot", "pot"]


def test_legacy_pot_recipes_normalize_to_scale_policy(apis, tmp_path):
    expected = _pylight_recipe(apis).modifiers[0].resolved_config
    recipe = _pylight_recipe(apis)
    group = recipe.modifiers[0].config_groups["group_linear"]
    for args in (group.weights, group.input_activations):
        args.format, args.scale_type = "int8_pot", "float"
    path = tmp_path / "legacy.yaml"
    recipe.yaml(str(path))
    loaded = apis.load_quantization_recipe(path)
    assert loaded.modifiers[0].resolved_config == expected


def test_static_int8_uses_the_same_pot_scale_policy(apis):
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(3))
    calibration = torch.tensor([[127.1, 2.5, 3.5]])
    recipe = apis.create_quantization_recipe("int8_static", scale_type="pot")
    apis.apply_quantization(model, recipe, calibrate=lambda: model(calibration))
    assert model[0].input_scale.item() == 2
    assert model[0].weight_scale.item() == 0.015625
    torch.testing.assert_close(model(calibration), torch.tensor([[128., 2., 4.]]), rtol=0, atol=0)


def test_pot_recipe_round_trip_and_runtime_updates(apis, tmp_path):
    recipe = _pylight_recipe(apis, targets=("Linear", "Conv2d"))
    path = tmp_path / "int8.yaml"
    recipe.yaml(str(path))
    loaded = apis.load_quantization_recipe(path)
    assert loaded.modifiers[0].resolved_config == recipe.modifiers[0].resolved_config
    assert not loaded.modifiers[0].resolved_config.requires_calibration_data()

    model = nn.Sequential(nn.Linear(7, 7, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(7))
    weights = model[0].weight.detach().clone()
    apis.apply_quantization(model, loaded)
    assert torch.equal(model[0].weight, weights)
    assert not hasattr(model[0], "weight_scale")
    assert not hasattr(model[0], "input_observer")

    # Halfway values use ties-to-even. amax=127 gives scale=1; just above
    # 127 gives scale=2, which catches accidentally using the divisor 127.5.
    x = torch.tensor([[-127., -3.5, -2.5, 0., 2.5, 3.5, 127.]])
    torch.testing.assert_close(model(x), torch.tensor([[-127., -4., -2., 0., 2., 4., 127.]]), rtol=0, atol=0)
    x[0, 0], x[0, -1] = -127.1, 127.1
    expected = torch.tensor([[-128., -4., -2., 0., 2., 4., 128.]])
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    # A later, larger batch must use a fresh range. Weights also remain live.
    with torch.no_grad():
        model[0].weight.mul_(4)
    torch.testing.assert_close(model(x * 16), expected * 64, rtol=0, atol=0)
    assert torch.equal(model[0].weight, weights * 4)
    assert model(torch.empty(0, 7)).shape == (0, 7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_and_noncontiguous_inputs(apis, device, dtype):
    forward = importlib.import_module("compressed_tensors.quantization.lifecycle.forward")
    args = _pylight_recipe(apis).modifiers[0].resolved_config.config_groups["group_linear"].input_activations
    x = torch.zeros(4, 7, device=device, dtype=dtype).t()
    scale, zp = forward.compute_dynamic_scales_and_zp(x, args, nn.Identity())
    assert scale.item() == 1 and zp.item() == 0
    result = forward.fake_quantize(x, scale, zp, args, None, None)
    assert result.dtype == dtype and result.device == x.device
    assert torch.equal(result, x)


@pytest.mark.parametrize("override", [
    {
        "num_bits": 7
    },
    {
        "type": "float"
    },
    {
        "symmetric": False
    },
    {
        "format": "mx9"
    },
    {
        "scale_dtype": torch.float32
    },
])
def test_incompatible_pot_recipe_rejected_before_conversion(apis, override):
    recipe = _pylight_recipe(apis)
    group = recipe.modifiers[0].config_groups["group_linear"]
    # A serialized custom recipe must fail too, not just the helper's presets.
    data = group.input_activations.model_dump()
    data.update(override)
    group.input_activations = type(group.input_activations).model_validate(data)
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="scale_type=pot requires"):
        apis.apply_quantization(model, recipe, calibrate=lambda: None)
    assert not hasattr(model[0], "quantization_scheme")


@pytest.mark.parametrize("scheme, options", [
    ("int8_dynamic", {}),
    ("int8_dynamic", {"scale_type": "pot", "activation_strategy": "tensor", "dynamic_weights": True}),
    ("int8_static", {}),
])
def test_conv2d_initialization_honours_ignore(apis, scheme, options):
    model = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1, groups=2), nn.Conv2d(4, 4, 1))
    reference = copy.deepcopy(model)
    x = torch.randn(2, 4, 5, 7)
    recipe = apis.create_quantization_recipe(scheme, targets=("Conv2d",), ignore=("1",), **options)
    kwargs = {"calibrate": lambda: model(x)} if scheme == "int8_static" else {}
    apis.apply_quantization(model, recipe, **kwargs)
    assert hasattr(model[0], "quantization_scheme")
    assert not hasattr(model[1], "quantization_scheme")
    output = model(x)
    assert torch.isfinite(output).all()
    assert not torch.equal(output, reference(x))
    assert torch.equal(model[1].weight, reference[1].weight)


@pytest.fixture
def pylight_int8():
    return pytest.importorskip("pytorchlight.quantize.core.int_ops")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_qparams_and_qdq_equal_pylight(apis, pylight_int8, device, dtype):
    forward = importlib.import_module("compressed_tensors.quantization.lifecycle.forward")
    args = _pylight_recipe(apis).modifiers[0].resolved_config.config_groups["group_linear"].input_activations
    # Reference settings: symmetric per-tensor INT8 with PoT scales.
    reference = pylight_int8.InputINTQuantizer(quant_scale_type="PoT", quant_dynamic_mode=True)
    torch.manual_seed(123)
    samples = [
        torch.zeros(3, 7),
        torch.tensor([-127.1, -127., -3.5, -2.5, 0., 2.5, 3.5, 127., 127.1]),
        torch.randn(3, 7, 5).transpose(0, 2),
        torch.rand(3, 7) * 1e-5,
        -torch.rand(3, 7) * 1024,
    ]
    for x in samples:
        x = x.to(device=device, dtype=dtype)
        reference.calculate_params(x)
        scale, zp = forward.compute_dynamic_scales_and_zp(x, args, nn.Identity())
        torch.testing.assert_close(scale.reshape(()), reference.interval, rtol=0, atol=0)
        assert zp.item() == reference.zero_point.item() == 0
        torch.testing.assert_close(forward.fake_quantize(x, scale, zp, args, None, None),
                                   reference.quantdequant(x),
                                   rtol=0,
                                   atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["linear", "conv2d", "grouped_conv2d"])
def test_layer_outputs_equal_pylight(apis, pylight_int8, device, dtype, kind):
    torch.manual_seed(12)
    if kind == "linear":
        layer, shape = nn.Linear(7, 5), (2, 3, 7)
    else:
        layer = nn.Conv2d(4, 4, 3, padding=2, dilation=2, groups=2 if kind == "grouped_conv2d" else 1)
        shape = (2, 4, 5, 7)
    model = nn.Sequential(layer.to(device=device, dtype=dtype)).eval()
    reference = copy.deepcopy(model[0])
    recipe = _pylight_recipe(apis, targets=("Linear", "Conv2d"))
    apis.apply_quantization(model, recipe)
    input_quantizer = pylight_int8.InputINTQuantizer(quant_scale_type="PoT", quant_dynamic_mode=True)
    weight_quantizer = pylight_int8.WeightINTQuantizer(quant_scale_type="PoT")
    original = model[0].weight.detach().clone()
    for factor in (0.01, 1., 100.):
        x = torch.randn(*shape, device=device, dtype=dtype) * factor
        input_quantizer.calculate_params(x)
        weight_quantizer.calculate_params(original)
        with torch.no_grad():
            reference.weight.copy_(weight_quantizer.quantdequant(original))
            expected = reference(input_quantizer.quantdequant(x))
            actual = model(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(model[0].weight, original)


def test_pot_and_mx_can_share_a_recipe(apis):
    if not torch.cuda.is_available():
        pytest.skip("MX kernels require a GPU")
    torch.manual_seed(9)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32)).cuda()
    reference = copy.deepcopy(model)
    recipe = _pylight_recipe(apis)
    modifier = recipe.modifiers[0]
    modifier.config_groups["group_linear"].targets = ["0"]
    mx = apis.create_quantization_recipe("mx6").modifiers[0].config_groups["group_linear"]
    mx.targets = ["1"]
    modifier.config_groups["mx_head"] = mx
    apis.apply_quantization(model, recipe)
    # Independent applications must give exactly the same mixed result.
    apis.apply_quantization(reference, _pylight_recipe(apis, ignore=("1",)))
    apis.apply_quantization(reference, apis.create_quantization_recipe("mx6", ignore=("0",)))
    x = torch.randn(2, 32, device="cuda")
    torch.testing.assert_close(model(x), reference(x), rtol=0, atol=0)
