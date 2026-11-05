import pytest
import torch

from alto.observers.hessian import HessianObserver
from alto.observers.per_channel_norm import PerChannelNormObserver


@pytest.mark.parametrize(
    "observer_cls",
    [HessianObserver, PerChannelNormObserver],
)
def test_stacked_linear_observer_uses_logical_input_dimension(observer_cls):
    module = torch.nn.Linear(5, 6)
    module.weight = torch.nn.Parameter(torch.empty(2, 3, 5))
    observer = observer_cls(base_name="test", module=module, device=torch.device("cpu"))
    inputs = torch.randn(2, 3, 4, 5)

    observer(inputs)

    rows = inputs.flatten(0, -2).to(torch.float32)
    if isinstance(observer, HessianObserver):
        assert observer.stats.shape == (5, 5)
        torch.testing.assert_close(observer.stats, rows.T @ rows)
    else:
        assert observer.stats.shape == (5,)
        torch.testing.assert_close(observer.stats, rows.square().sum(dim=0) / 2)
