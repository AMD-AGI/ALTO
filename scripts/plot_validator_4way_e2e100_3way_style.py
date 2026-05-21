#!/usr/bin/env python3
"""4-way validator-on E2E (100-step) loss / grad comparison, rendered in the
same single-panel-loss + triple-panel-diagnostic layout that we use for
the long-horizon 3-way 1000-step figures (see
``plot_longhorizon_3way_e2e1000_loss_curves.py``).

Mirrors ``doc/nvfp4_knowledge_transfer/design/e2e_loss_plot_format_zh.md``:
top loss line chart + Setup row + bold "Recipe methodology" header +
monospace-aligned 14-knob recipe table.

Recipes compared (sweep stamp ``validator_4way_20260519_070723``):
  * BF16  baseline
  * MXFP4  default
  * NVFP4  default              (spec-fix, grouped_sr=inherit)
  * NVFP4-test-2                (spec-fix, ignore first-1 + last-2)

Outputs (under ``doc/nvfp4_knowledge_transfer/reports/``):
  * ``validator_4way_e2e100step_loss.png``             — single panel,
        per-step training loss vs step (100 points / recipe).
  * ``validator_4way_e2e100step_loss_val_grad.png``    — triple panel:
        validation loss / training loss / log-scale pre-clip grad_norm.

The existing ``validator_4way_e2e100step_val_loss.png`` and
``validator_4way_e2e100step_val_train_grad.png`` produced by
``plot_validator_4way_e2e100_curves.py`` are left untouched so that
historical references remain valid.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAMP = "validator_4way_20260519_070723"
LOG_DIR = Path("/wekafs/hanwang2/workspace/alto_runs/orchestrator_logs")
LOGS = {
    "bf16":          LOG_DIR / f"e2e100_val_bf16_{STAMP}.log",
    "mxfp4":         LOG_DIR / f"e2e100_val_mxfp4_{STAMP}.log",
    "nvfp4_default": LOG_DIR / f"e2e100_val_nvfp4_default_{STAMP}.log",
    "nvfp4_test_2":  LOG_DIR / f"e2e100_val_nvfp4_test_2_{STAMP}.log",
}

OUT_DIR = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports")
OUT_LOSS = OUT_DIR / "validator_4way_e2e100step_loss.png"
OUT_ALL = OUT_DIR / "validator_4way_e2e100step_loss_val_grad.png"


LABEL = {
    "bf16":          "BF16  baseline",
    "mxfp4":         "MXFP4  default (w 2D / a,g 1D + RHT + SR)",
    "nvfp4_default": "NVFP4  default  (spec-fix, grouped_sr=inherit, ignore last-3)",
    "nvfp4_test_2":  "NVFP4-test-2  (spec-fix, grouped_sr=inherit, ignore first-1 + last-2)",
}
COLOR = {
    "bf16":          "#1f4fa1",
    "mxfp4":         "#1f9d55",
    "nvfp4_default": "#d6322f",
    "nvfp4_test_2":  "#e76f51",
}
LINESTYLE = {
    "bf16":          "-",
    "mxfp4":         "--",
    "nvfp4_default": "-",
    "nvfp4_test_2":  "-",
}
ORDER = ["bf16", "mxfp4", "nvfp4_default", "nvfp4_test_2"]


RECIPE_TABLE_ROWS = [
    ("knob",                "MXFP4 default",                "NVFP4 default",                       "NVFP4-test-2"),
    ("recipe yaml",         "lpt_recipe.yaml",              "lpt_nvfp4_20b_r8_recipe.yaml",        "lpt_nvfp4_20b_first1_last2_bf16_recipe.yaml"),
    ("scheme",              "mxfp4",                        "nvfp4",                               "nvfp4"),
    ("block_size",          "32",                           "16",                                  "16"),
    ("block scale format",  "E8M0 (FP8 exponent)",          "E4M3 (FP8 normal)",                   "E4M3 (FP8 normal)"),
    ("two_level_scaling",   "none (no PTS)",                "tensorwise (FP32 PTS, spec-fix)",     "tensorwise (FP32 PTS, spec-fix)"),
    ("use_2dblock_w",       "true  (w block: 32x32)",       "true  (w block: 16x16)",              "true  (w block: 16x16)"),
    ("use_2dblock_x",       "false (a,g 1D: 1x32)",         "false (a,g 1D: 1x16)",                "false (a,g 1D: 1x16)"),
    ("use_hadamard (RHT)",  "true  (on wgrad path)",        "true  (on wgrad path)",               "true  (on wgrad path)"),
    ("use_sr_grad",         "true  (SR on backward)",       "true  (SR on backward)",              "true  (SR on backward)"),
    ("grouped_sr_mode",     "n/a (no grouped knob)",        "inherit  (SR on dgrad+wgrad)",        "inherit  (SR on dgrad+wgrad)"),
    ("use_dge",             "false",                        "false",                               "false"),
    ("clip_mode",           "none",                         "none",                                "none"),
    ("ignore",              "output, router.gate",          "output, router.gate, layers.21-23",   "output, router.gate, layers.0, layers.22-23"),
    ("targets",             "Linear, GptOssGroupedExperts", "Linear, GptOssGroupedExperts",        "Linear, GptOssGroupedExperts"),
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


def _fmt_delta(x: float) -> str:
    """Pretty-print a signed delta (used in the final val_loss table)."""
    sign = "+" if x >= 0 else "−"  # use proper minus sign for readability
    return f"{sign}{abs(x):.4f}"


SHORT_LABEL = {
    "bf16":          "BF16",
    "mxfp4":         "MXFP4 default",
    "nvfp4_default": "NVFP4 default",
    "nvfp4_test_2":  "NVFP4-test-2",
}


def build_final_val_loss_table(series, baseline_tag: str = "bf16",
                               default_tag: str = "nvfp4_default"):
    """Build the Final-validation-loss summary table required by spec §1.5.

    Uses short recipe labels (``SHORT_LABEL``) so the table fits side-by-side
    with the wide recipe-knob table.  Long descriptive legend text already
    appears in the chart legend; this summary table is the numeric headline.

    Returns (table_text, final_step) where final_step is the latest val step
    common to all recipes (so the headline number is apples-to-apples).
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
    "100 train steps, warmup auto-collapsed to 100 (probe), no checkpoint, "
    "validator.freq=10, validator.steps=10, "
    "val dataset = c4-validation-91205-samples (held-out, fixed across all 4 runs). "
    "training.dtype unchanged across all four runs."
)

METHODOLOGY = "Recipe methodology (MXFP4 default vs NVFP4 default vs NVFP4-test-2):"


def plot_loss_only(series):
    """Single-panel per-step training loss vs step (primary deliverable)."""
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
            linestyle=LINESTYLE[tag], linewidth=2.0,
        )
    ax.set_title(
        "GPT-OSS-20B · 100-step E2E training loss · "
        "BF16 vs MXFP4-default vs NVFP4-default vs NVFP4-test-2",
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
    """Triple panel: val loss / train loss / grad_norm (log-scale).

    Bottom strip is two side-by-side blocks:
        LEFT  — Recipe methodology header + 16-row knob × recipe table
        RIGHT — Final-validation-loss summary header (spec §1.5) + summary
                table with short recipe labels (4 columns; Δ vs BF16 and
                Δ vs NVFP4-default)
    """
    fig, (ax_val, ax_train, ax_grad) = plt.subplots(1, 3, figsize=(22, 9.2))
    for tag in ORDER:
        val = series[tag]["val"]
        train = series[tag]["train"]
        if val:
            steps = sorted(val)
            ax_val.plot(
                steps, [val[s] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=2.0, marker="o", markersize=4,
            )
        if train:
            steps = sorted(train)
            ax_train.plot(
                steps, [train[s][0] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.6,
            )
            ax_grad.plot(
                steps, [train[s][1] for s in steps],
                label=LABEL[tag], color=COLOR[tag],
                linestyle=LINESTYLE[tag], linewidth=1.4,
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
    ax_grad.axhline(1.0, color="gray", linestyle=":", linewidth=1.0,
                    label="clip threshold (1.0)")
    ax_grad.grid(alpha=0.3, which="both")
    ax_grad.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle(
        "GPT-OSS-20B · 100-step E2E (validator-on) · "
        "BF16 / MXFP4-default / NVFP4-default / NVFP4-test-2 on 8 × MI300X",
        fontsize=12,
    )
    summary, final_step = build_final_val_loss_table(series)
    # LEFT block: recipe knob table
    fig.text(0.005, 0.330, METHODOLOGY,
             fontsize=9, color="#000", weight="bold")
    fig.text(0.005, 0.005, RECIPE_TABLE,
             fontsize=7.0, family="monospace", color="#222")
    # RIGHT block: final validation loss summary (per spec §1.5)
    fig.text(0.71, 0.330,
             f"Final validation loss summary (step {final_step}):",
             fontsize=9, color="#000", weight="bold")
    fig.text(0.71, 0.245, summary,
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
        print(f"  {tag:>16s}: val_n={len(val):<4d} train_n={len(train):<4d} src={path.name}")
        series[tag] = {"val": val, "train": train}
    plot_loss_only(series)
    plot_all(series)


if __name__ == "__main__":
    main()
