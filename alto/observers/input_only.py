# modified from https://github.com/vllm-project/llm-compressor/blob/bede809f388aaeb1438a4d692d2d79109f9357dc/src/llmcompressor/modifiers/pruning/sparsegpt/sgpt_sparsify.py
# licensed under the Apache License 2.0

import torch
from alto.utils.pytorch.module import TransformerConv1D
from .base import Observer, register_observer

PRECISION = torch.bfloat16


@register_observer("input_only")
class InputOnlyObserver(Observer):
    """
    Input-only observer: collects the raw layer inputs during calibration.
    Each forward input is concatenated along dim=0 (batch), e.g. single input
    [1, 32, 2048] -> after N calls stats is [N_total, 32, 2048]. No reshape/transpose.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        self.register_buffer("num_samples", torch.zeros([1], dtype=torch.int32, device=self.device))
        # Placeholder with numel()==0 so first forward assigns; shape unknown until first input
        self.register_buffer("stats", torch.zeros(0, dtype=PRECISION, device=self.device))

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        inp = x_orig.detach().to(PRECISION)
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)  # [B, C] -> [1, B, C] so batch is always dim=0
        num_added = inp.shape[0]
        self.num_samples += num_added
        if self.stats.numel() == 0:
            self.stats = inp
        else:
            self.stats = torch.cat([self.stats, inp], dim=0)

        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.num_samples.zero_()
            self.stats = torch.zeros(0, dtype=PRECISION, device=self.device)

    def calculate_params(self):
        pass
