# modified from https://github.com/vllm-project/llm-compressor/blob/bede809f388aaeb1438a4d692d2d79109f9357dc/src/llmcompressor/modifiers/pruning/sparsegpt/sgpt_sparsify.py
# licensed under the Apache License 2.0

import torch
from modeloptimizer.utils.pytorch.module import TransformerConv1D
from .base import Observer, register_observer

PRECISION = torch.float32


@register_observer("hessian")
class HessianObserver(Observer):
    """
    Hessian matrix of the activations.
        H = \sum_{i=1}^{N} x_i x_i^T / N.
        A naive implementation of the Hessian matrix.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["should_calculate_gparam"] = False
        kwargs["should_calculate_qparams"] = False
        super().__init__(*args, **kwargs)
        self.register_buffer("stats", self.make_empty_stats())
        self.register_buffer("num_samples", torch.zeros([1], dtype=torch.int32, device=self.device))

    def make_empty_stats(self) -> torch.Tensor:
        weight = self.module().weight
        num_columns = weight.shape[1]
        return torch.zeros((num_columns, num_columns), dtype=PRECISION, device=self.device)

    def get_current_min_max(self, observed: torch.Tensor):
        pass

    def get_current_global_min_max(self, observed: torch.Tensor):
        pass

    def reshape_input(self, inp: torch.Tensor) -> torch.Tensor:
        module = self.module()
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        num_added = inp.shape[0]
        if isinstance(module, torch.nn.Linear) or (TransformerConv1D and isinstance(module, TransformerConv1D)):
            if inp.dim() == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(module, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        return inp, num_added

    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        with torch.no_grad():
            inp = x_orig.detach()
            inp, num_added = self.reshape_input(inp)
            self.num_samples += num_added
            inp = inp.type(PRECISION)
            self.stats += inp.matmul(inp.t())

        return x_orig

    def clear_stats(self):
        with torch.no_grad():
            self.stats.zero_()
            self.num_samples.zero_()

    def calculate_params(self):
        pass


@register_observer("hessian_obs")
class HessianObsObserver(HessianObserver):
    """
    Hessian matrix of the activations.
        A normalized implementation of the Hessian matrix.
        Sample-by-sample amplitude decay with increasing sample size.
        Suitable for SparseGPT and GPTQ.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


    def forward_inner(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig

        with torch.no_grad():
            inp = x_orig.detach()
            inp, num_added = self.reshape_input(inp)
            self.stats *= self.num_samples / (self.num_samples + num_added)
            self.num_samples += num_added
            inp = (2 / self.num_samples) ** 0.5 * inp.type(PRECISION)
            self.stats += inp.matmul(inp.t())

        return x_orig
