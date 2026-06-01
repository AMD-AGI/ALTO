#!/usr/bin/env python3
"""4-way validator-enabled E2E comparison plot for the
2026-05-19 NVFP4-test-2 sweep.

Follows the standard format in
``doc/nvfp4_knowledge_transfer/design/e2e_loss_plot_format_zh.md``:
top **validation loss** line chart (11 points per recipe @ step 1, 10, ..., 100)
+ setup row + bold "Recipe methodology" header + monospace recipe knob table.

Outputs:
  * ``validator_4way_e2e100step_val_loss.png`` — single panel val loss vs step
  * ``validator_4way_e2e100step_val_train_grad.png`` — triple panel: val loss /
    train loss / log-scale grad_norm
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAMP = "validator_4way_20260519_070723"
LOG_DIR = Path("/wekafs/zhitwang/alto_runs/orchestrator_logs")
LOGS = {
    "bf16": LOG_DIR / f"e2e100_val_bf16_{STAMP}.log",
    "mxfp4": LOG_DIR / f"e2e100_val_mxfp4_{STAMP}.log",
    "nvfp4_default": LOG_DIR / f"e2e100_val_nvfp4_default_{STAMP}.log",
    "nvfp4_test_2": LOG_DIR / f"e2e100_val_nvfp4_test_2_{STAMP}.log",
}
OUT_DIR = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports")
OUT_VAL = OUT_DIR / "validator_4way_e2e100step_val_loss.png"
OUT_ALL = OUT_DIR / "validator_4way_e2e100step_val_train_grad.png"

LABEL = {
    "bf16": "BF16  baseline",
    "mxfp4": "MXFP4  default (w 2D / a,g 1D + RHT + SR)",
    "nvfp4_default": "NVFP4  default  (spec-fix, grouped_sr=inherit, ignore last-3)",
    "nvfp4_test_2": "NVFP4-test-2  (spec-fix, grouped_sr=inherit, ignore first-1 + last-2)",
}
COLOR = {
    "bf16": "#1f4fa1",
    "mxfp4": "#1f9d55",
    "nvfp4_default": "#d6322f",
    "nvfp4_test_2": "#e76f51",
}
LINESTYLE = {
    "bf16": "-",
    "mxfp4": "--",
    "nvfp4_default": "-",
    "nvfp4_test_2": "-",
}
MARKER = {
    "bf16": "o",
    "mxfp4": "s",
    "nvfp4_default": "^",
    "nvfp4_test_2": "D",
}
ORDER = ["bf16", "mxfp4", "nvfp4_default", "nvfp4_test_2"]


RECIPE_TABLE_ROWS = [
    ("knob",                "MXFP4 default",                "NVFP4 default",                       "NVFP4-test-2"),
    ("recipe yaml",         "lpt_recipe.yaml",              "lpt_nvfp4_20b_r8_recipe.yaml",         "lpt_nvfp4_20b_first1_last2_bf16_recipe.yaml"),
    ("scheme",              "mxfp4",                        "nvfp4",                                "nvfp4"),
    ("block_size",          "32",                           "16",                                   "16"),
    ("block scale format",  "E8M0 (FP8 exponent)",          "E4M3 (FP8 normal)",                    "E4M3 (FP8 normal)"),
    ("two_level_scaling",   "none (no PTS)",                "tensorwise (FP32 PTS, spec-fix)",      "tensorwise (FP32 PTS, spec-fix)"),
    ("use_2dblock_w",       "true  (w block: 32x32)",       "true  (w block: 16x16)",               "true  (w block: 16x16)"),
    ("use_2dblock_x",       "false (a,g 1D: 1x32)",         "false (a,g 1D: 1x16)",                 "false (a,g 1D: 1x16)"),
    ("use_hadamard (RHT)",  "true  (on wgrad path)",        "true  (on wgrad path)",                "true  (on wgrad path)"),
    ("use_sr_grad",         "true  (SR on backward)",       "true  (SR on backward)",               "true  (SR on backward)"),
    ("grouped_sr_mode",     "n/a (no grouped knob)",        "inherit  (SR on dgrad+wgrad)",         "inherit  (SR on dgrad+wgrad)"),
    ("use_dge",             "false",                        "false",                                "false"),
    ("clip_mode",           "none",                         "none",                                 "none"),
    ("ignore",              "output, router.gate",          "output, router.gate, layers.21-23",    "output, router.gate, layers.0, layers.22-23"),
    ("targets",             "Linear, GptOssGroupedExperts", "Linear, GptOssGroupedExperts",         "Linear, GptOssGroupedExperts"),
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
VAL_PAT = re.compile(r"validate step:\s*(\d+).*?loss:\s*([0-9.]+)")
TRAIN_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.eE+-]+)"
)


def load(path: Path):
    val: dict[int, float] = {}
    train: dict[int, tuple[float, float]] = {}
    if not path.exists():
        return val, train
    for raw in path.read_text(errors="ignore").splitlines():
        s = ANSI.sub("", raw)
        m = VAL_PAT.search(s)
        if m:
            val[int(m.group(1))] = float(m.group(2))
            continue
        m2 = TRAIN_PAT.search(s)
        if m2:
            train[int(m2.group(1))] = (float(m2.group(2)), float(m2.group(3)))
    return val, train


def plot_val_only(series):
    fig, ax = plt.subplots(figsize=(14, 9.2))
    for tag in ORDER:
        val = series[tag]["val"]
        if not val:
            continue
        steps = sorted(val)
        losses = [val[s] for s in steps]
        ax.plot(
            steps,
            losses,
            label=LABEL[tag],
            color=COLOR[tag],
            linestyle=LINESTYLE[tag],
            linewidth=2.0,
            marker=MARKER[tag],
            markersize=5,
        )
    ax.set_title(
        "GPT-OSS-20B · 100-step E2E · validation loss · "
        "BF16 vs MXFP4-default vs NVFP4-default vs NVFP4-test-2",
        fontsize=12,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_xlim(left=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.text(
        0.01,
        0.345,
        "Setup: gpt-oss-20b, C4 megatron en, seq_len=8192, GBS=16, FSDP=2 TP=EP=4 ETP=1, seed=42, "
        "warmup auto-collapsed to 100, no checkpoint, validator.freq=10, validator.steps=10, "
        "val dataset = c4-validation-91205-samples (held-out, fixed across all 4 runs). "
        "training.dtype unchanged across all four runs.",
        fontsize=7,
        color="#555",
    )
    fig.text(
        0.01,
        0.310,
        "Recipe methodology (MXFP4 default vs NVFP4 default vs NVFP4-test-2):",
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
    fig.savefig(OUT_VAL, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_VAL}")


def plot_all(series):
    fig, (ax_val, ax_train, ax_grad) = plt.subplots(1, 3, figsize=(22, 9.2))
    for tag in ORDER:
        val = series[tag]["val"]
        train = series[tag]["train"]
        if val:
            steps = sorted(val)
            ax_val.plot(
                steps, [val[s] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=2.0,
                marker=MARKER[tag], markersize=5,
            )
        if train:
            steps = sorted(train)
            ax_train.plot(
                steps, [train[s][0] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.5,
            )
            ax_grad.plot(
                steps, [train[s][1] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.5,
            )
    ax_val.set_title("Validation loss vs step (11 points)")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("validation loss")
    ax_val.set_xlim(left=1)
    ax_val.grid(alpha=0.3)
    ax_val.legend(loc="upper right", fontsize=8, frameon=True)

    ax_train.set_title("Per-step training loss vs step (100 points)")
    ax_train.set_xlabel("step")
    ax_train.set_ylabel("per-step training loss")
    ax_train.set_xlim(left=1)
    ax_train.grid(alpha=0.3)
    ax_train.legend(loc="upper right", fontsize=8, frameon=True)

    ax_grad.set_title("Pre-clip grad_norm vs step (log)")
    ax_grad.set_xlabel("step")
    ax_grad.set_ylabel("grad_norm  (clip threshold = 1.0)")
    ax_grad.set_yscale("log")
    ax_grad.set_xlim(left=1)
    ax_grad.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="clip threshold (1.0)")
    ax_grad.grid(alpha=0.3, which="both")
    ax_grad.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "GPT-OSS-20B · 100-step E2E (validator-on) · "
        "BF16 / MXFP4-default / NVFP4-default / NVFP4-test-2 on 8 × MI300X",
        fontsize=12,
    )
    fig.text(
        0.005,
        0.330,
        "Recipe methodology (MXFP4 default vs NVFP4 default vs NVFP4-test-2):",
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
    fig.savefig(OUT_ALL, dpi=170, bbox_inches="tight")
    print(f"Saved {OUT_ALL}")


def main():
    series = {}
    for tag, path in LOGS.items():
        val, train = load(path)
        series[tag] = {"val": val, "train": train}
    plot_val_only(series)
    plot_all(series)


if __name__ == "__main__":
    main()
