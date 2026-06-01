#!/usr/bin/env python3
"""NVFP4-test-2 16K-step training: validation-loss curve.

Single-panel val-loss-vs-step plot for the completed 16,000-step
NVFP4-test-2 GPT-OSS-20B run (GBS=16).  Data merged from the early-phase
status cache (steps 1-5000) and the run's train log (steps 5500-16000).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/alto/doc/nvfp4_knowledge_transfer/reports/"
           "nvfp4_test2_16k_val_loss_curve.png")

# Validation loss (step -> loss) for the full 16K run.
VAL = {
    1: 11.3904, 500: 4.8605, 1000: 4.4142, 1500: 4.2047, 2000: 4.1287,
    2500: 4.0581, 3000: 3.9961, 3500: 3.9831, 4000: 3.9051, 4500: 3.8283,
    5000: 3.8301, 5500: 3.8103, 6000: 3.8295, 6500: 3.7502, 7000: 3.7261,
    7500: 3.7156, 8000: 3.7178, 8500: 3.7082, 9000: 3.7336, 9500: 3.6764,
    10000: 3.6502, 10500: 3.6259, 11000: 3.6580, 11500: 3.6437, 12000: 3.6229,
    12500: 3.5744, 13000: 3.6018, 13500: 3.5648, 14000: 3.5613, 14500: 3.5596,
    15000: 3.5734, 15500: 3.5766, 16000: 3.6116,
}

MLPERF_TARGET = 3.34
BEST_STEP, BEST_VAL = 14500, 3.5596


def main():
    steps = sorted(VAL)
    vals = [VAL[s] for s in steps]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(steps, vals, color="#e76f51", linewidth=1.8, marker="o",
            markersize=3.5, label="NVFP4-test-2  val loss  (GBS=16)")

    # MLPerf target line
    ax.axhline(MLPERF_TARGET, color="#1f9d55", linestyle="--", linewidth=1.3,
               label=f"MLPerf target = {MLPERF_TARGET}")
    # best-val marker
    ax.scatter([BEST_STEP], [BEST_VAL], color="#1f4fa1", zorder=5, s=60,
               marker="*", label=f"best val {BEST_VAL:.4f} @ step {BEST_STEP}")
    ax.annotate(f"best {BEST_VAL:.4f}\n@ {BEST_STEP}",
                (BEST_STEP, BEST_VAL), textcoords="offset points",
                xytext=(8, 18), fontsize=8, color="#1f4fa1")
    # final point
    ax.annotate(f"final {VAL[16000]:.4f}\n@ 16000",
                (16000, VAL[16000]), textcoords="offset points",
                xytext=(-10, 22), fontsize=8, color="#b5651d")

    ax.set_title("GPT-OSS-20B · NVFP4-test-2 · 16,000-step validation loss\n"
                 "(NVFP4 + 2D-block-w + Hadamard + SR-grad + tensorwise; "
                 "first-1 + last-2 layers BF16; GBS=16, LR=4e-4)",
                 fontsize=11)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    ax.set_xlim(0, 16200)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)

    note = ("Val plateaus ~3.56 (best @ step 14500) then rises to 3.61 @ 16000 "
            "— small-batch (GBS=16) plateau well above the 3.34 target; the "
            "late rise indicates near-constant LR / mild overfit at the tail.")
    fig.text(0.01, 0.005, note, fontsize=7.5, color="#555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
