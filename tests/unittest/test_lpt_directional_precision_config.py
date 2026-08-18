import pytest

from alto.modifiers.lpt.base import LowPrecisionTrainingModifier


@pytest.mark.parametrize(
    "forward_precision,backward_precision",
    [("low", "low"), ("bf16", "low"), ("low", "bf16"), ("bf16", "bf16")],
)
def test_directional_precision_config(
    forward_precision: str, backward_precision: str
) -> None:
    modifier = LowPrecisionTrainingModifier(
        scheme="mxfp4",
        forward_precision=forward_precision,
        backward_precision=backward_precision,
    )
    config = next(iter(modifier.resolved_config))
    assert config.forward_precision == forward_precision
    assert config.backward_precision == backward_precision
