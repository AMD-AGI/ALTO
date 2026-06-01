#!/usr/bin/env python3
"""Plot the spec-fix 100-step E2E loss / grad_norm curves for
BF16 / MXFP4 / NVFP4 on GPT-OSS-20B.

Output:
  * doc/nvfp4_knowledge_transfer/reports/specfix_e2e100_loss.png
      single-panel TorchBench / mlperf-style training loss vs step
  * doc/nvfp4_knowledge_transfer/reports/specfix_e2e100_loss_grad.png
      two-panel diagnostic (loss + grad_norm, log scale)

Data source: summary CSV produced by the 100-step driver
    e2e100_summary_<stamp>.csv
with columns:
    step,loss_bf16,grad_bf16,loss_mxfp4,grad_mxfp4,loss_nvfp4,grad_nvfp4
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = Path("/wekafs/zhitwang/alto_runs/orchestrator_logs")
SUMMARY_CSV = LOG_DIR / "e2e100_summary_specfix_e2e_100step_20260518_143205.csv"
OUT_DIR = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports")
OUT_LOSS = OUT_DIR / "specfix_e2e100_loss.png"
OUT_LOSS_GRAD = OUT_DIR / "specfix_e2e100_loss_grad.png"

LABEL = {
    "bf16": "BF16  baseline",
    "mxfp4": "MXFP4  current best recipe (w 2D / a,g 1D + RHT + SR)",
    "nvfp4": "NVFP4  spec-fix (PTS / quant order aligned with NVFP4 spec)",
}
COLOR = {
    "bf16": "#1f4fa1",
    "mxfp4": "#1f9d55",
    "nvfp4": "#d6322f",
}
LINESTYLE = {
    "bf16": "-",
    "mxfp4": "--",
    "nvfp4": "-",
}
ORDER = ["bf16", "mxfp4", "nvfp4"]


def load(csv_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    series: dict[str, dict[int, tuple[float, float]]] = {t: {} for t in ORDER}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                continue
            for tag in ORDER:
                lv = (row.get(f"loss_{tag}", "") or "").strip()
                gv = (row.get(f"grad_{tag}", "") or "").strip()
                if lv and gv and lv != "None" and gv != "None":
                    try:
                        series[tag][step] = (float(lv), float(gv))
                    except ValueError:
                        pass
    return series


def plot_loss_only(series: dict[str, dict[int, tuple[float, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for tag in ORDER:
        rows = series.get(tag, {})
        if not rows:
            continue
        steps = sorted(rows)
        losses = [rows[s][0] for s in steps]
        ax.plot(
            steps,
            losses,
            label=LABEL[tag],
            color=COLOR[tag],
            linestyle=LINESTYLE[tag],
            linewidth=2.0,
        )
    ax.set_title(
        "GPT-OSS-20B · 100-step E2E training loss · "
        "BF16 vs MXFP4 vs NVFP4 (PTS spec-fix)",
        fontsize=12,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_xlim(left=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.text(
        0.01,
        0.01,
        "Setup: hf_assets=gpt-oss-20b, dataset=C4 megatron en, "
        "seq_len=8192, GBS=16, FSDP=2, TP=EP=4, ETP=1, seed=42, "
        "warmup auto-collapsed to 100 (probe run, no checkpoint/validator). "
        "BF16/MXFP4 recipes unchanged; NVFP4 uses default recipe "
        "(lpt_nvfp4_20b_r8_recipe.yaml) after spec-aligned PTS / quant order.",
        fontsize=7,
        color="#555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT_LOSS, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_LOSS}")


def plot_loss_grad(series: dict[str, dict[int, tuple[float, float]]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6))
    for tag in ORDER:
        rows = series.get(tag, {})
        if not rows:
            continue
        steps = sorted(rows)
        losses = [rows[s][0] for s in steps]
        grads = [rows[s][1] for s in steps]
        ax1.plot(
            steps,
            losses,
            label=LABEL[tag],
            color=COLOR[tag],
            linestyle=LINESTYLE[tag],
            linewidth=2.0,
        )
        ax2.plot(
            steps,
            grads,
            label=LABEL[tag],
            color=COLOR[tag],
            linestyle=LINESTYLE[tag],
            linewidth=2.0,
        )

    ax1.set_title("Training loss vs step")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")
    ax1.set_xlim(left=1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9, frameon=True)

    ax2.set_title("Pre-clip grad_norm vs step (log scale)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("grad_norm  (clip threshold = 1.0)")
    ax2.set_yscale("log")
    ax2.set_xlim(left=1)
    ax2.axhline(
        1.0,
        color="gray",
        linestyle=":",
        linewidth=1.0,
        label="clip threshold (1.0)",
    )
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=9, frameon=True)

    fig.suptitle(
        "GPT-OSS-20B · 100-step E2E (spec-fix NVFP4) · "
        "BF16 / MXFP4 / NVFP4 on 8 × MI300X",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT_LOSS_GRAD, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_LOSS_GRAD}")


def main() -> None:
    series = load(SUMMARY_CSV)
    if not all(series[t] for t in ORDER):
        missing = [t for t in ORDER if not series[t]]
        raise SystemExit(
            f"Missing series in {SUMMARY_CSV}: {missing}"
        )
    plot_loss_only(series)
    plot_loss_grad(series)


if __name__ == "__main__":
    main()
