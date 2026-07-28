#!/usr/bin/env python3
"""Plot step vs. validation loss from one or more slurm .out logs.

Parses lines like:
    ... validate step: 768  loss:  4.9572  memory: ...
(ANSI color codes are stripped before matching.)

Alongside the plot, a table on the right lists each method's loss at every
step (union of all steps across files; blank where a method has no datapoint).

Usage:
    python3 plot_val_loss.py <logfile> [<logfile> ...] [-o output.png]

A custom legend/column name can be given per file with "path=label" syntax, e.g.:
    python3 plot_val_loss.py run_a.out=baseline run_b.out="lr 3e-4"
Files given without "=label" fall back to their basename.
"""
import argparse
import os
import re
import sys
import textwrap

import matplotlib.pyplot as plt

# strip ANSI escape sequences, then pull step + loss
ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")


def parse(path):
    steps, losses = [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = VAL.search(ANSI.sub("", line))
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfiles", nargs="+",
                    help='one or more slurm .out logs; use "path=label" to set a custom legend name')
    ap.add_argument("-o", "--output", help="output image path (default: val_loss.png)")
    args = ap.parse_args()

    out = args.output or "val_loss.png"

    # figure: plot on the left, table on the right
    fig, (ax, ax_tbl) = plt.subplots(
        1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 1.4]}
    )

    labels = []                 # column labels, in input order
    loss_by_step = {}           # label -> {step: loss}
    colors = {}                 # label -> line color
    total = 0

    for entry in args.logfiles:
        # allow "path=custom legend label"; only split on the first "="
        if "=" in entry:
            path, label = entry.split("=", 1)
        else:
            path, label = entry, os.path.basename(entry)

        steps, losses = parse(path)
        if not steps:
            print(f"warning: no validation datapoints found in {path}", file=sys.stderr)
            continue
        total += len(steps)

        (line,) = ax.plot(steps, losses, marker="o", linewidth=1.5, label=label)
        labels.append(label)
        loss_by_step[label] = dict(zip(steps, losses))
        colors[label] = line.get_color()

    if total == 0:
        sys.exit("No validation datapoints found in any input file")

    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Validation loss on GPT-OSS 20B")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # --- table of losses per step for each method ---
    all_steps = sorted({s for d in loss_by_step.values() for s in d})
    # wrap long method names so they don't overflow their table column
    wrapped = ["\n".join(textwrap.wrap(lab, width=14)) or lab for lab in labels]
    col_labels = ["step"] + wrapped
    cell_text = []
    for s in all_steps:
        row = [str(s)]
        for lab in labels:
            v = loss_by_step[lab].get(s)
            row.append(f"{v:.4f}" if v is not None else "")
        cell_text.append(row)

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # give the header row enough height for the tallest wrapped label
    max_lines = max(lbl.count("\n") + 1 for lbl in col_labels)
    base_h = table[0, 0].get_height()
    # color the header cells to match each method's line
    for c, (raw, lab) in enumerate(zip(["step"] + labels, col_labels)):
        cell = table[0, c]
        cell.set_height(base_h * max_lines)
        cell.set_text_props(weight="bold")
        if raw in colors:
            cell.set_facecolor(colors[raw])
            cell.set_text_props(weight="bold", color="white")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Wrote {out} ({total} validation datapoints across {len(args.logfiles)} file(s))")


if __name__ == "__main__":
    main()
