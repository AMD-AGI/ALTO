#!/usr/bin/env python3
"""Sweep5 GBS=64 val-loss comparison (completed/partial runs).

Plots validation loss vs step for the GBS=64 quick-validation runs:
  c3 BF16 baseline, c7 all-layers NVFP4 (rank0), c1 NVFP4-test-2 (first1+last2).
Reads directly from the run logs so partial (still-running) runs are included.
"""
from __future__ import annotations
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = Path("/wekafs/zhitwang/alto_runs/orchestrator_logs")
OUT = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports/"
           "sweep5_gbs64_val_comparison.png")

RUNS = {
    "c3 BF16 baseline (GBS=64)":
        "c3_bf16_gbs64_20260531_161417.log",
    "c7 all-layers NVFP4 (GBS=64, rank0)":
        "c7_gbs64_comp_off_all_20260531_195544.log",
    "c1 NVFP4-test-2 first1+last2 (GBS=64)":
        "c1_nvfp4_test2_gbs64_20260601_015520.log",
}
COLOR = {
    "c3 BF16 baseline (GBS=64)": "#1f4fa1",
    "c7 all-layers NVFP4 (GBS=64, rank0)": "#1f9d55",
    "c1 NVFP4-test-2 first1+last2 (GBS=64)": "#e76f51",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"validate step:\s*(\d+).*?loss:\s*([0-9.]+)")


def load_val(path: Path):
    val = {}
    if not path.exists():
        return val
    for raw in path.read_text(errors="ignore").splitlines():
        if "[rank0]" not in raw:
            continue
        m = VAL.search(ANSI.sub("", raw))
        if m:
            val[int(m.group(1))] = float(m.group(2))
    return val


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, fname in RUNS.items():
        val = load_val(LOG_DIR / fname)
        if not val:
            print(f"[warn] no val for {fname}")
            continue
        steps = sorted(val)
        ax.plot(steps, [val[s] for s in steps], marker="o", markersize=4,
                linewidth=1.8, color=COLOR[label],
                label=f"{label}  [{max(steps)} steps]")
        print(f"{label}: {[(s, round(val[s],4)) for s in steps]}")

    ax.set_title("GPT-OSS-20B · GBS=64 quick-val sweep · validation loss vs step\n"
                 "(500-step runs from HF init; c1 still in progress)",
                 fontsize=11)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    note = ("At GBS=64 the three recipes (BF16 / all-layers-NVFP4 / "
            "NVFP4-test-2) track within ~0.05 val loss through step 300-500 "
            "— layer policy & NVFP4 quantization barely matter this early. "
            "All-layers NVFP4 (c7) completed 500 steps without divergence. "
            "c5 (rank-32 outlier compensation) is omitted: it crashes at model "
            "build (DecomposedLinear meta-device bug).")
    fig.text(0.01, 0.005, note, fontsize=7.5, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
