# AWQ activation scale observer.
# Collects per-channel mean of |x|, matching awq/quantize/auto_scale.py:get_act_scale

import torch
from .base import Observer, register_observer

PRECISION = torch.float32


@register_observer("activation_scale")
class ActivationScaleObserver(Observer):
    """Collects per-channel mean of absolute activation values.

    Equivalent to AWQ's ``get_act_scale(x) = x.abs().view(-1, x.shape[-1]).mean(0)``,
    accumulated as a running mean across calibration batches.

    After calibration, ``self.stats`` holds the per-channel activation scale
    and ``self.num_samples`` holds the total number of token vectors observed.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        # stats buffer will be lazily initialized on first forward call
        self.stats = None
        self.num_samples = torch.zeros(tuple(), dtype=PRECISION, device=self.device)

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        with torch.no_grad():
            x = x_orig.detach().float()
            flat = x.abs().view(-1, x.shape[-1])  # [tokens, hidden]
            batch_mean = flat.mean(0)
            count = flat.shape[0]

            # Lazy init stats on first call
            if self.stats is None:
                self.stats = torch.zeros_like(batch_mean)

            # Incremental mean update
            total = self.num_samples + count
            self.stats.mul_(self.num_samples / total.clamp(min=1))
            self.stats.add_(batch_mean * (count / total.clamp(min=1)))
            self.num_samples.add_(count)

        return x_orig

    def clear_stats(self):
        self.stats = None
        self.num_samples.zero_()

    def calculate_params(self):
        pass
