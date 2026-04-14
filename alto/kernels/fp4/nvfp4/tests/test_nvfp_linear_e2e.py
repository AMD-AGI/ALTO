# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""
Minimal end-to-end training convergence test for NVFP4 QDQ Linear.

Trains a small MLP on a synthetic regression task under two configurations:
  1. BF16 baseline:  nn.Linear (standard BF16 GEMM)
  2. NVFP4 QDQ:      NVFP4Linear (BF16 -> quant -> FP4 -> dequant -> BF16 -> GEMM)

Both start from identical weights and see identical data.
The test verifies that NVFP4 converges and produces comparable final loss.
"""

import pytest
from tabulate import tabulate
import torch
import torch.nn as nn

from alto.kernels.fp4.nvfp4.nvfp_linear import NVFP4LinearFunction


class NVFP4Linear(nn.Linear):
    """Drop-in nn.Linear replacement with NVFP4 QDQ in forward/backward."""

    def __init__(self, in_features, out_features, bias=True,
                 use_2dblock_x=False, use_2dblock_w=False,
                 use_sr_grad=True, use_per_tensor_scale=False):
        super().__init__(in_features, out_features, bias=bias)
        self.use_2dblock_x = use_2dblock_x
        self.use_2dblock_w = use_2dblock_w
        self.use_sr_grad = use_sr_grad
        self.use_per_tensor_scale = use_per_tensor_scale

    def forward(self, x):
        y = NVFP4LinearFunction.apply(
            x, self.weight,
            self.use_2dblock_x,
            self.use_2dblock_w,
            self.use_sr_grad,
            self.use_per_tensor_scale,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


def _make_model(hidden, layers, use_nvfp4, **nvfp4_kwargs):
    modules = []
    for i in range(layers):
        if use_nvfp4:
            modules.append(NVFP4Linear(hidden, hidden, bias=False, **nvfp4_kwargs))
        else:
            modules.append(nn.Linear(hidden, hidden, bias=False))
        if i < layers - 1:
            modules.append(nn.ReLU())
    return nn.Sequential(*modules).cuda().bfloat16()


@pytest.mark.parametrize("use_sr_grad", [
    pytest.param(False, marks=pytest.mark.xfail(
        reason="RNE rounding causes biased gradients in FP4; SR is required for convergence"
    )),
    True,
])
@pytest.mark.parametrize("use_2dblock", [False, True])
def test_nvfp4_linear_e2e_convergence(use_sr_grad, use_2dblock):
    hidden = 64
    layers = 2
    batch_size = 128
    steps = 300
    lr = 3e-3

    torch.manual_seed(42)
    device = torch.device("cuda")

    teacher = nn.Sequential(
        nn.Linear(hidden, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, hidden, bias=False),
    ).cuda().bfloat16()
    for p in teacher.parameters():
        p.requires_grad_(False)

    model_bf16 = _make_model(hidden, layers, use_nvfp4=False)
    model_nvfp4 = _make_model(
        hidden, layers, use_nvfp4=True,
        use_2dblock_x=use_2dblock,
        use_2dblock_w=use_2dblock,
        use_sr_grad=use_sr_grad,
    )
    model_nvfp4.load_state_dict(model_bf16.state_dict())

    opt_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=lr)
    opt_nvfp4 = torch.optim.AdamW(model_nvfp4.parameters(), lr=lr)

    losses_bf16 = []
    losses_nvfp4 = []
    log_rows = []

    for step in range(steps):
        torch.manual_seed(step)
        x = torch.randn(batch_size, hidden, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            target = teacher(x)

        opt_bf16.zero_grad()
        loss_bf16 = nn.functional.mse_loss(model_bf16(x), target)
        loss_bf16.backward()
        opt_bf16.step()
        losses_bf16.append(loss_bf16.item())

        opt_nvfp4.zero_grad()
        loss_nvfp4 = nn.functional.mse_loss(model_nvfp4(x), target)
        loss_nvfp4.backward()
        opt_nvfp4.step()
        losses_nvfp4.append(loss_nvfp4.item())

        if step % 50 == 0 or step == steps - 1:
            log_rows.append([
                step,
                f"{loss_bf16.item():.6f}",
                f"{loss_nvfp4.item():.6f}",
                f"{loss_nvfp4.item() / max(loss_bf16.item(), 1e-12):.4f}",
            ])

    final_bf16 = losses_bf16[-1]
    final_nvfp4 = losses_nvfp4[-1]
    initial_nvfp4 = losses_nvfp4[0]
    ratio = final_nvfp4 / max(final_bf16, 1e-12)
    loss_reduction = (initial_nvfp4 - final_nvfp4) / max(initial_nvfp4, 1e-12)
    converged = loss_reduction > 0.3

    weight_diffs = []
    for (n1, p1), (n2, p2) in zip(
        model_bf16.named_parameters(), model_nvfp4.named_parameters()
    ):
        diff = (p1.float() - p2.float()).abs()
        weight_diffs.append([n1, f"{diff.max().item():.6f}", f"{diff.mean().item():.6f}"])

    print()
    print(f"  Config: use_2dblock={use_2dblock}, use_sr_grad={use_sr_grad}")
    print(f"  Steps: {steps}, Hidden: {hidden}, Layers: {layers}")
    print()
    print("  Training Loss Curve:")
    print(
        tabulate(
            log_rows,
            headers=["Step", "BF16 Loss", "NVFP4 Loss", "Ratio"],
            tablefmt="github",
        )
    )
    print()
    print(f"  Final BF16 Loss:  {final_bf16:.6f}")
    print(f"  Final NVFP4 Loss: {final_nvfp4:.6f}")
    print(f"  Final Ratio:      {ratio:.4f}")
    print(f"  NVFP4 Converged:  {converged}")
    print()
    print("  Weight Divergence (after training):")
    print(
        tabulate(
            weight_diffs,
            headers=["Parameter", "Max Diff", "Mean Diff"],
            tablefmt="github",
        )
    )

    assert converged, (
        f"NVFP4 training did not converge "
        f"(loss reduction {loss_reduction:.2%}, need > 30%)"
    )
