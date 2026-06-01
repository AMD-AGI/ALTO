#!/usr/bin/env python3
"""3-way long-horizon E2E (1000-step) loss / grad comparison.

Compares:
  * BF16 baseline           (longhorizon_1k_20260520_025621 stamp)
  * NVFP4 default           (longhorizon_1k_20260520_025621 stamp)
  * NVFP4-test-2 (retry)    (retry_nodbg_20260520_142532 stamp — the
                             clean-finish 1000-step run launched after
                             a 10-min cool-down; the prior nvfp4_test_2
                             1000-step run aborted at step 43 due to a
                             transient SIGABRT on rank 2, see
                             ``nvfp4_test_2_longhorizon_failure_zh.md``)

Follows ``doc/nvfp4_knowledge_transfer/design/e2e_loss_plot_format_zh.md``:
top loss / val-loss line chart + setup row + bold "Recipe methodology"
header + monospace-aligned 14-knob recipe table.

Outputs (under ``doc/nvfp4_knowledge_transfer/reports/``):
  * ``longhorizon_3way_e2e1000step_loss.png``               — single panel,
        per-step training loss vs step (1000 points / recipe)
  * ``longhorizon_3way_e2e1000step_val_train_grad.png``     — triple panel:
        validation loss / training loss / log-scale pre-clip grad_norm
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_DIR = Path("/wekafs/zhitwang/alto_runs/orchestrator_logs")
LONGHORIZON_STAMP = "longhorizon_1k_20260520_025621"
RETRY_STAMP = "retry_nodbg_20260520_142532"

LOGS = {
    "bf16":          LOG_DIR / f"e2e1k_longhorizon_bf16_{LONGHORIZON_STAMP}.log",
    "nvfp4_default": LOG_DIR / f"e2e1k_longhorizon_nvfp4_default_{LONGHORIZON_STAMP}.log",
    "nvfp4_test_2":  LOG_DIR / f"e2e1k_retry_no_debug_nvfp4_test_2_{RETRY_STAMP}.log",
}

OUT_DIR = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports")
OUT_LOSS = OUT_DIR / "longhorizon_3way_e2e1000step_loss.png"
OUT_ALL = OUT_DIR / "longhorizon_3way_e2e1000step_val_train_grad.png"

LABEL = {
    "bf16":          "BF16  baseline",
    "nvfp4_default": "NVFP4  default  (spec-fix, grouped_sr=inherit, ignore last-3)",
    "nvfp4_test_2":  "NVFP4-test-2  (spec-fix, grouped_sr=inherit, ignore first-1 + last-2)",
}
COLOR = {
    "bf16":          "#1f4fa1",
    "nvfp4_default": "#d6322f",
    "nvfp4_test_2":  "#e76f51",
}
LINESTYLE = {
    "bf16":          "-",
    "nvfp4_default": "-",
    "nvfp4_test_2":  "-",
}
ORDER = ["bf16", "nvfp4_default", "nvfp4_test_2"]


RECIPE_TABLE_ROWS = [
    ("knob",                "NVFP4 default",                       "NVFP4-test-2"),
    ("recipe yaml",         "lpt_nvfp4_20b_r8_recipe.yaml",        "lpt_nvfp4_20b_first1_last2_bf16_recipe.yaml"),
    ("scheme",              "nvfp4",                               "nvfp4"),
    ("block_size",          "16",                                  "16"),
    ("block scale format",  "E4M3 (FP8 normal)",                   "E4M3 (FP8 normal)"),
    ("two_level_scaling",   "tensorwise (FP32 PTS, spec-fix)",     "tensorwise (FP32 PTS, spec-fix)"),
    ("use_2dblock_w",       "true  (w block: 16x16)",              "true  (w block: 16x16)"),
    ("use_2dblock_x",       "false (a,g 1D: 1x16)",                "false (a,g 1D: 1x16)"),
    ("use_hadamard (RHT)",  "true  (on wgrad path)",               "true  (on wgrad path)"),
    ("use_sr_grad",         "true  (SR on backward)",              "true  (SR on backward)"),
    ("grouped_sr_mode",     "inherit  (SR on dgrad+wgrad)",        "inherit  (SR on dgrad+wgrad)"),
    ("use_dge",             "false",                               "false"),
    ("clip_mode",           "none",                                "none"),
    ("ignore",              "output, router.gate, layers.21-23",   "output, router.gate, layers.0, layers.22-23"),
    ("targets",             "Linear, GptOssGroupedExperts",        "Linear, GptOssGroupedExperts"),
]


def _format_table(rows) -> str:
    cols = list(zip(*rows))
    widths = [max(len(s) for s in col) for col in cols]
    lines = []
    for i, row in enumerate(rows):
        cells = [row[j].ljust(widths[j]) for j in range(len(row))]
        lines.append(" | ".join(cells))
        if i == 0:
            lines.append("-+-".join("-" * w for w in widths))
    return "\n".join(lines)


RECIPE_TABLE = _format_table(RECIPE_TABLE_ROWS)


SHORT_LABEL = {
    "bf16":          "BF16",
    "nvfp4_default": "NVFP4 default",
    "nvfp4_test_2":  "NVFP4-test-2",
}


def _fmt_delta(x: float) -> str:
    """Pretty-print a signed delta (used in the final val_loss table)."""
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x):.4f}"


def build_final_val_loss_table(series, baseline_tag: str = "bf16",
                               default_tag: str = "nvfp4_default"):
    """Spec §1.5 final-validation-loss summary table.

    Returns (table_text, final_step) where final_step is the latest validate
    step that ALL recipes have, so the headline numbers are apples-to-apples.
    """
    common = None
    for tag, s in series.items():
        if not s["val"]:
            continue
        steps = set(s["val"])
        common = steps if common is None else (common & steps)
    if not common:
        return "(no validation data)", None
    final_step = max(common)
    base = series[baseline_tag]["val"].get(final_step) if baseline_tag in series else None
    default = series[default_tag]["val"].get(final_step) if default_tag in series else None
    rows = [("recipe", "final val_loss", "Δ vs BF16", "Δ vs NVFP4-default")]
    for tag in ORDER:
        v = series[tag]["val"].get(final_step)
        if v is None:
            continue
        d_b = _fmt_delta(v - base) if (base is not None and tag != baseline_tag) else "(baseline)"
        d_d = _fmt_delta(v - default) if (default is not None and tag != default_tag) else "(self)"
        if tag == baseline_tag:
            d_b = "(baseline)"
        if tag == default_tag:
            d_d = "(self)"
        rows.append((SHORT_LABEL.get(tag, tag), f"{v:.4f}", d_b, d_d))
    return _format_table(rows), final_step


ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL_PAT = re.compile(r"validate step:\s*(\d+).*?loss:\s*([0-9.]+)")
TRAIN_PAT = re.compile(
    r"^(?:\[rank0\]:)?\[titan\][^|]*?step:\s*(\d+)\s+loss:\s*([0-9.]+)\s+grad_norm:\s*([0-9.eE+-]+)"
)


def load(path: Path):
    val: dict[int, float] = {}
    train: dict[int, tuple[float, float]] = {}
    if not path.exists():
        print(f"[warn] log not found: {path}")
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


SETUP_LINE = (
    "Setup: gpt-oss-20b, C4 megatron en, seq_len=8192, GBS=16, FSDP=2 TP=EP=4 ETP=1, seed=42, "
    "1000 train steps, warmup=128, no checkpoint, validator.freq=10, validator.steps=10, "
    "val dataset = c4-validation-91205-samples (held-out, fixed across all 3 runs). "
    "NVFP4-test-2 is the 10-min cool-down retry that completed cleanly after a transient "
    "rank-2 SIGABRT killed the previous attempt at step 43. "
    "training.dtype unchanged across all three runs."
)

METHODOLOGY = "Recipe methodology (NVFP4 default vs NVFP4-test-2):"


def plot_loss_only(series):
    """Single-panel per-step training loss vs step."""
    fig, ax = plt.subplots(figsize=(14, 9.2))
    for tag in ORDER:
        train = series[tag]["train"]
        if not train:
            continue
        steps = sorted(train)
        losses = [train[s][0] for s in steps]
        ax.plot(
            steps, losses,
            label=LABEL[tag], color=COLOR[tag],
            linestyle=LINESTYLE[tag], linewidth=1.6,
        )
    ax.set_title(
        "GPT-OSS-20B · 1000-step E2E training loss · "
        "BF16 vs NVFP4-default vs NVFP4-test-2",
        fontsize=12,
    )
    ax.set_xlabel("step")
    ax.set_ylabel("per-step training loss")
    ax.set_xlim(left=1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.text(0.01, 0.345, SETUP_LINE, fontsize=7, color="#555")
    fig.text(0.01, 0.310, METHODOLOGY, fontsize=9, color="#000", weight="bold")
    fig.text(0.01, 0.005, RECIPE_TABLE, fontsize=7.0, family="monospace", color="#222")
    fig.tight_layout(rect=(0, 0.38, 1, 1))
    fig.savefig(OUT_LOSS, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_LOSS}")


def plot_all(series):
    """Triple panel: val loss / train loss / grad_norm (log)."""
    fig, (ax_val, ax_train, ax_grad) = plt.subplots(1, 3, figsize=(22, 9.2))
    for tag in ORDER:
        val = series[tag]["val"]
        train = series[tag]["train"]
        if val:
            steps = sorted(val)
            ax_val.plot(
                steps, [val[s] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=2.0, marker="o", markersize=3,
            )
        if train:
            steps = sorted(train)
            ax_train.plot(
                steps, [train[s][0] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.4,
            )
            ax_grad.plot(
                steps, [train[s][1] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.2,
            )
    ax_val.set_title("Validation loss vs step (101 points)")
    ax_val.set_xlabel("step")
    ax_val.set_ylabel("validation loss")
    ax_val.set_xlim(left=1)
    ax_val.grid(alpha=0.3)
    ax_val.legend(loc="upper right", fontsize=8, frameon=True)

    ax_train.set_title("Per-step training loss vs step (1000 points)")
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
    ax_grad.axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
                    label="clip threshold (1.0)")
    ax_grad.grid(alpha=0.3, which="both")
    ax_grad.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "GPT-OSS-20B · 1000-step E2E (validator-on) · "
        "BF16 / NVFP4-default / NVFP4-test-2 on 8 × MI300X",
        fontsize=12,
    )
    summary, final_step = build_final_val_loss_table(series)
    # LEFT block: recipe knob table
    fig.text(0.005, 0.330, METHODOLOGY,
             fontsize=9, color="#000", weight="bold")
    fig.text(0.005, 0.005, RECIPE_TABLE,
             fontsize=7.0, family="monospace", color="#222")
    # RIGHT block: final validation loss summary (spec §1.5)
    fig.text(0.71, 0.330,
             f"Final validation loss summary (step {final_step}):",
             fontsize=9, color="#000", weight="bold")
    fig.text(0.71, 0.260, summary,
             fontsize=7.5, family="monospace", color="#222")
    fig.tight_layout(rect=(0, 0.38, 1, 0.95))
    fig.savefig(OUT_ALL, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_ALL}")
    print()
    print(f"Final validation loss summary (step {final_step}):")
    print(summary)


def main():
    series = {}
    for tag, path in LOGS.items():
        val, train = load(path)
        print(f"  {tag:>16s}: val_n={len(val):<4d} train_n={len(train):<5d} src={path.name}")
        series[tag] = {"val": val, "train": train}
    plot_loss_only(series)
    plot_all(series)


if __name__ == "__main__":
    main()
