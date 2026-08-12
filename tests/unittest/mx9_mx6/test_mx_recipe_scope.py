# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Scoping an MX recipe with ``ignore``, and the guards around it.

The model is a plain three-stage stack with normalization layers interleaved, not
any real architecture, because the API must not know about one.

Run with:
    pytest tests/unittest/mx9_mx6/test_mx_recipe_scope.py
"""

import pytest
import torch
import torch.nn as nn

from alto.models.petr.quantize import apply_mx_quantization


class Stack(nn.Module):
    """Three named parts, each mixing quantizable leaves with normalization."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.LayerNorm(64),
            nn.Linear(64, 16, bias=False),
        )

    def forward(self, x):
        x = self.trunk(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the MX codec is a Triton kernel and needs a GPU",
)

pytestmark = requires_cuda


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Stack().cuda().to(torch.bfloat16).eval()


@pytest.fixture
def inputs():
    generator = torch.Generator(device="cpu").manual_seed(1)
    return torch.randn(2, 16, 8, 8, generator=generator).cuda().to(torch.bfloat16)


def quantized_names(model):
    return {
        name
        for name, module in model.named_modules()
        if getattr(module, "quantization_scheme", None) is not None
    }


def test_uniform_recipe_covers_every_leaf(model, inputs):
    apply_mx_quantization(model, "mx6")

    assert quantized_names(model) == {"stem.0", "trunk.0", "head.0", "head.2"}
    with torch.no_grad():
        model(inputs)


def test_normalization_is_never_quantized(model):
    """Normalization owns a ``weight`` and would be quantized if ever targeted."""
    apply_mx_quantization(model, "mx6")

    for name in ("stem.1", "trunk.1", "head.1"):
        assert getattr(model.get_submodule(name), "quantization_scheme", None) is None


def test_ignored_modules_keep_their_dtype(model, inputs):
    """The BF16 half: no scheme attached, weights bit-for-bit untouched."""
    head_weight = model.head[0].weight.detach().clone()

    apply_mx_quantization(model, "mx6", ignore=["re:^head\\..*"])

    assert quantized_names(model) == {"stem.0", "trunk.0"}
    assert getattr(model.head[0], "quantization_scheme", None) is None
    assert torch.equal(model.head[0].weight, head_weight)
    assert model.head[0].weight.dtype == torch.bfloat16
    with torch.no_grad():
        model(inputs)  # the mixed graph still runs


def test_bare_name_covers_the_subtree(model):
    """compressed_tensors alone would match the container, which holds no leaf."""
    apply_mx_quantization(model, "mx9", ignore=["head"])

    assert quantized_names(model) == {"stem.0", "trunk.0"}


def test_bare_name_can_also_be_a_single_leaf(model):
    apply_mx_quantization(model, "mx9", ignore=["head.0", "head.2"])

    assert quantized_names(model) == {"stem.0", "trunk.0"}


def test_bare_name_does_not_match_a_prefix_of_a_sibling(model):
    """``stem`` must not also swallow a hypothetical ``stem_extra``."""
    torch.manual_seed(0)
    wider = Stack().cuda().to(torch.bfloat16).eval()
    wider.add_module("stem_extra", nn.Conv2d(16, 16, 1, bias=False).cuda().to(torch.bfloat16))

    apply_mx_quantization(wider, "mx6", ignore=["stem"])

    assert "stem_extra" in quantized_names(wider)
    assert "stem.0" not in quantized_names(wider)


def test_ignore_that_excludes_nothing_is_rejected(model):
    """A typo'd pattern would otherwise quantize the whole model silently."""
    with pytest.raises(ValueError, match="excluded nothing"):
        apply_mx_quantization(model, "mx6", ignore=["no_such_part"])


def test_recipe_that_matches_nothing_is_rejected():
    model = nn.Sequential(nn.BatchNorm2d(8), nn.ReLU()).cuda()
    with pytest.raises(ValueError, match="matched no"):
        apply_mx_quantization(model, "mx6")


def test_unsupported_format_is_rejected(model):
    """The format is validated while the recipe is built, before the scope is."""
    with pytest.raises(ValueError, match="unsupported MX format"):
        apply_mx_quantization(model, "mx7")

    assert quantized_names(model) == set()


def test_scheme_carries_the_mx_format(model):
    """``format`` is what the patched fake_quantize dispatches the codec on."""
    apply_mx_quantization(model, "mx9", ignore=["re:^head\\..*"])

    scheme = model.stem[0].quantization_scheme
    assert scheme.weights.format == "mx9"
    assert scheme.input_activations.format == "mx9"
    assert getattr(model.head[0], "quantization_scheme", None) is None


def test_a_rejected_ignore_leaves_the_model_untouched(model):
    """Scope is validated before anything is wrapped, so the failure is atomic."""
    with pytest.raises(ValueError, match="excluded nothing"):
        apply_mx_quantization(model, "mx6", ignore=["no_such_part"])

    assert quantized_names(model) == set()


def test_block_axis_reaches_the_kernel_per_module_type(model, inputs, monkeypatch):
    """The axis in the scheme has to arrive at the codec, or blocks run the wrong way.

    Conv2d tensors are 4D here and must block along axis 1, their input channels;
    Linear tensors are 2D and block along the last axis. Blocking a 3x3 weight
    along the last axis instead would put three elements in a 16-element block.
    """
    import alto.kernels.mx as mx

    seen = []
    convert_to_mx = mx.convert_to_mx

    def spy(data_hp, target_dtype, axis=-1):
        seen.append((tuple(data_hp.shape), axis))
        return convert_to_mx(data_hp, target_dtype, axis=axis)

    monkeypatch.setattr(mx, "convert_to_mx", spy)

    apply_mx_quantization(model, "mx6")
    with torch.no_grad():
        model(inputs)

    assert {(len(shape), axis) for shape, axis in seen} == {(4, 1), (2, -1)}
    assert (tuple(model.stem[0].weight.shape), 1) in seen
    assert (tuple(model.head[0].weight.shape), -1) in seen


@pytest.mark.xfail(
    strict=True,
    reason="known gap: freeze_module_quantization only promotes COMPRESSED to "
    "FROZEN, and a fully-dynamic recipe never becomes COMPRESSED, so finalize "
    "leaves the status at CALIBRATION. Drop this marker once that is fixed.",
)
def test_finalized_modules_are_frozen(model):
    """A finished lifecycle should not still advertise itself as calibrating."""
    from compressed_tensors.quantization import QuantizationStatus

    apply_mx_quantization(model, "mx6")

    assert model.stem[0].quantization_status == QuantizationStatus.FROZEN


def test_scoped_run_stays_closer_to_the_unquantized_model(model, inputs):
    """Quantizing fewer layers has to move the output less, or ignore is inert."""
    with torch.no_grad():
        reference = model(inputs).clone()

    torch.manual_seed(0)
    uniform = Stack().cuda().to(torch.bfloat16).eval()
    apply_mx_quantization(uniform, "mx6")

    torch.manual_seed(0)
    scoped = Stack().cuda().to(torch.bfloat16).eval()
    apply_mx_quantization(scoped, "mx6", ignore=["re:^head\\..*"])

    with torch.no_grad():
        uniform_out = uniform(inputs)
        scoped_out = scoped(inputs)

    assert not torch.equal(uniform_out, scoped_out)
    assert (scoped_out - reference).norm() < (uniform_out - reference).norm()
