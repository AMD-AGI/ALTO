#!/usr/bin/env python3
"""3-way E2E loss / grad_norm comparison plot: BF16 / MXFP4 default / NVFP4 default.

Follows the standard format in
``doc/nvfp4_knowledge_transfer/design/e2e_loss_plot_format_zh.md``:
top loss line chart + setup row + bold "Recipe methodology" header +
monospace-aligned recipe knob table.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = Path("/wekafs/hanwang2/workspace/alto_runs/orchestrator_logs")
LOGS = {
    "bf16": LOG_DIR / "e2e100_bf16_specfix_e2e_100step_20260518_143205.log",
    "mxfp4": LOG_DIR / "e2e100_mxfp4_specfix_e2e_100step_20260518_143205.log",
    "nvfp4_default": LOG_DIR / "e2e100_nvfp4_specfix_e2e_100step_20260518_143205.log",
}
OUT_DIR = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports")
OUT_LOSS = OUT_DIR / "specfix_3way_e2e100step_loss.png"
OUT_LOSS_GRAD = OUT_DIR / "specfix_3way_e2e100step_loss_grad.png"

LABEL = {
    "bf16": "BF16  baseline",
    "mxfp4": "MXFP4  default (w 2D / a,g 1D + RHT + SR)",
    "nvfp4_default": "NVFP4  default  (spec-fix, grouped_sr=inherit)",
}
COLOR = {
    "bf16": "#1f4fa1",
    "mxfp4": "#1f9d55",
    "nvfp4_default": "#d6322f",
}
LINESTYLE = {
    "bf16": "-",
    "mxfp4": "--",
    "nvfp4_default": "-",
}
ORDER = ["bf16", "mxfp4", "nvfp4_default"]


RECIPE_TABLE_ROWS = [
    ("knob",                "MXFP4 default",                "NVFP4 default"),
    ("recipe yaml",         "lpt_recipe.yaml",              "lpt_nvfp4_20b_r8_recipe.yaml"),
    ("scheme",              "mxfp4",                        "nvfp4"),
    ("block_size",          "32",                           "16"),
    ("block scale format",  "E8M0 (FP8 exponent)",          "E4M3 (FP8 normal)"),
    ("two_level_scaling",   "none (no PTS)",                "tensorwise (FP32 PTS, spec-fix)"),
    ("use_2dblock_w",       "true  (w block: 32x32)",       "true  (w block: 16x16)"),
    ("use_2dblock_x",       "false (a,g 1D: 1x32)",         "false (a,g 1D: 1x16)"),
    ("use_hadamard (RHT)",  "true  (on wgrad path)",        "true  (on wgrad path)"),
    ("use_sr_grad",         "true  (SR on backward)",       "true  (SR on backward)"),
    ("grouped_sr_mode",     "n/a (no grouped knob)",        "inherit  (SR on dgrad+wgrad)"),
    ("use_dge",             "false",                        "false"),
    ("clip_mode",           "none",                         "none"),
    ("ignore",              "output, router.gate",          "output, router.gate, layers.21-23"),
    ("targets",             "Linear, GptOssGroupedExperts", "Linear, GptOssGroupedExperts"),
]


def _format_recipe_table() -> str:
    cols = list(zip(*RECIPE_TABLE_ROWS))
    widths = [max(len(s) for s in col) for col in cols]
    lines = []
    for i, row in enumerate(RECIPE_TABLE_ROWS):
        cells = [row[j].ljust(widths[j]) for j in range(len(row))]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


RECIPE_TABLE = _format_recipe_table()


ANSI = re.compile(r"\x1b\[[0-9;]*m")
PAT = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.]+).*?grad_norm:\s*([0-9.eE+-]+)")


def load(path: Path) -> dict[int, tuple[float, float]]:
    rows: dict[int, tuple[float, float]] = {}
    if not path.exists():
        return rows
    for raw in path.read_text(errors="ignore").splitlines():
        s = ANSI.sub("", raw)
        if "validate step" in s:
            continue  # skip val rows in this 3-way (no-validator) historical run
        m = PAT.search(s)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return rows


def plot_loss_only(series: dict[str, dict[int, tuple[float, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 9.2))
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
        "BF16 vs MXFP4-default vs NVFP4-default (spec-fix)",
        fontsize=12,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_xlim(left=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.text(
        0.01,
        0.345,
        "Setup: gpt-oss-20b, C4 megatron en, seq_len=8192, GBS=16, FSDP=2 TP=EP=4 ETP=1, seed=42, "
        "warmup auto-collapsed to 100 (probe, no checkpoint/validator). "
        "training.dtype unchanged across all three runs.",
        fontsize=7,
        color="#555",
    )
    fig.text(
        0.01,
        0.310,
        "Recipe methodology (MXFP4 default vs NVFP4 default):",
        fontsize=9,
        color="#000",
        weight="bold",
    )
    fig.text(
        0.01,
        0.005,
        RECIPE_TABLE,
        fontsize=7.0,
        family="monospace",
        color="#222",
    )
    fig.tight_layout(rect=(0, 0.38, 1, 1))
    fig.savefig(OUT_LOSS, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_LOSS}")


def plot_loss_grad(series: dict[str, dict[int, tuple[float, float]]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9.2))
    for tag in ORDER:
        rows = series.get(tag, {})
        if not rows:
            continue
        steps = sorted(rows)
        losses = [rows[s][0] for s in steps]
        grads = [rows[s][1] for s in steps]
        ax1.plot(steps, losses, label=LABEL[tag], color=COLOR[tag],
                 linestyle=LINESTYLE[tag], linewidth=2.0)
        ax2.plot(steps, grads, label=LABEL[tag], color=COLOR[tag],
                 linestyle=LINESTYLE[tag], linewidth=2.0)
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
        "GPT-OSS-20B · 100-step E2E · BF16 / MXFP4-default / NVFP4-default on 8 × MI300X",
        fontsize=12,
    )
    fig.text(
        0.005,
        0.330,
        "Recipe methodology (MXFP4 default vs NVFP4 default):",
        fontsize=9,
        color="#000",
        weight="bold",
    )
    fig.text(
        0.005,
        0.005,
        RECIPE_TABLE,
        fontsize=7.0,
        family="monospace",
        color="#222",
    )
    fig.tight_layout(rect=(0, 0.38, 1, 0.95))
    fig.savefig(OUT_LOSS_GRAD, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_LOSS_GRAD}")


def main() -> None:
    series = {k: load(p) for k, p in LOGS.items()}
    plot_loss_only(series)
    plot_loss_grad(series)


if __name__ == "__main__":
    main()
