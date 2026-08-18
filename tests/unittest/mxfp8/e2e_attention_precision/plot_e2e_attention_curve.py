# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT
"""Run and plot toy-attention E2E experiments.

This is a standalone helper, not a pytest test:

    python tests/unittest/mxfp8/plot_e2e_attention_curve.py

By default it follows the existing Toy-MoE style: one deterministic seed, longer
training, and plain loss curves. It still writes raw loss series to JSON so the
figure can be audited later.
"""

import argparse
from datetime import datetime
import json
import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e2e_attention_common import (  # noqa: E402
        COLORS,
        DEFAULT_SEEDS,
        PATHS,
        AttentionExperimentConfig,
        run_forward_only_experiment,
        run_training_experiment,
    )
else:
    from .e2e_attention_common import (  # noqa: E402
        COLORS,
        DEFAULT_SEEDS,
        PATHS,
        AttentionExperimentConfig,
        run_forward_only_experiment,
        run_training_experiment,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--output-dir", default=None,
                        help="Directory for outputs. Defaults to a unique run directory.")
    parser.add_argument("--run-name", default=None,
                        help="Name under e2e_attention_runs when --output-dir is not set.")
    parser.add_argument("--show-std-band", action="store_true",
                        help="Draw mean +/- std shading when running multiple seeds.")
    return parser.parse_args()


def _resolve_output_dir(args):
    if args.output_dir is not None:
        return args.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_part = f"seed{DEFAULT_SEEDS[0]}" if args.num_seeds == 1 else f"seeds{args.num_seeds}"
    run_name = args.run_name or f"{timestamp}_steps{args.steps}_{seed_part}"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_attention_runs", run_name)


def _save_json(payload, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"saved results to {out_path}")


def _plot_panel(ax, title, result, show_std_band):
    steps = range(result["config"]["steps"])
    for name in PATHS:
        data = result["paths"][name]
        mean = data["mean"]
        std = data["std"]
        ax.plot(
            steps,
            mean,
            label=name,
            color=COLORS[name],
            linestyle="--" if name == "bf16 baseline" else "-",
        )
        if show_std_band:
            lower = [max(m - s, 1e-12) for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            ax.fill_between(steps, lower, upper, color=COLORS[name], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def _save_plot(training_result, forward_result, out_path, show_std_band):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("\nmatplotlib is not installed; skipped saving curve image.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    suffix = ", mean +/- std" if show_std_band else ""
    _plot_panel(axes[0], f"independent training{suffix}", training_result, show_std_band)
    _plot_panel(axes[1], f"bf16 trajectory forward-only{suffix}", forward_result, show_std_band)
    fig.suptitle("toy attention training")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved curve to {out_path}")


def _print_summary(title, result):
    print(f"\n===== {title} =====")
    print(f"{'path':>24} {'final mean':>14} {'final std':>14}")
    for name in PATHS:
        data = result["paths"][name]
        print(f"{name:>24} {data['final_mean']:>14.8f} {data['final_std']:>14.8f}")


def main():
    args = _parse_args()
    seeds = DEFAULT_SEEDS[:args.num_seeds]
    config = AttentionExperimentConfig(steps=args.steps)
    device = torch.device("cuda")

    print(
        "\n===== toy attention config: "
        f"batch={config.batch}, seqlen={config.seqlen}, "
        f"heads={config.num_heads}, head_dim={config.head_dim}, "
        f"steps={config.steps}, seeds={list(seeds)} ====="
    )

    training_result = run_training_experiment(config, seeds=seeds, device=device)
    forward_result = run_forward_only_experiment(config, seeds=seeds, device=device)
    payload = {
        "training": training_result,
        "forward_only_bf16_trajectory": forward_result,
    }

    output_dir = _resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)
    print(f"output directory: {output_dir}")
    json_path = os.path.join(output_dir, "e2e_attention_results.json")
    png_path = os.path.join(output_dir, "e2e_attention_loss_curve.png")
    _save_json(payload, json_path)
    _save_plot(training_result, forward_result, png_path, args.show_std_band)
    _print_summary("independent training", training_result)
    _print_summary("bf16 trajectory forward-only", forward_result)


if __name__ == "__main__":
    main()
