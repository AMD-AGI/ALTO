#!/usr/bin/env python3
"""Plot step vs. validation loss from one or more slurm .out logs.

Parses lines like:
    ... validate step: 768  loss:  4.9572  memory: ...
(ANSI color codes are stripped before matching.)

Alongside the plot, a table on the right lists each method's loss at every
step (union of all steps across files; blank where a method has no datapoint).

Usage:
    # inline on the command line
    python3 plot_training_stats.py <logfile> [<logfile> ...] [-o output.png]
    # or from a .toml config
    python3 plot_training_stats.py -c runs.toml

A custom legend/column name can be given per file with "path=label" syntax, e.g.:
    python3 plot_training_stats.py run_a.out=baseline run_b.out="lr 3e-4"
Files given without "=label" fall back to their basename.

A single run may span several .out files (e.g. a resumed job). List them
comma-separated on the CLI; they are parsed in order and drawn as one curve:
    python3 plot_training_stats.py part1.out,part2.out=baseline

TOML config format (see runs.example.toml):
    output = "val_loss.png"                 # optional, overridden by -o
    title  = "Validation loss vs. step"     # optional

    [[runs]]
    file = "slurm-207639.out"
    name = "baseline"

    [[runs]]
    file = "slurm-207504.out"
    name = "lr 3e-4"

    # a run split across multiple files: pass a list to "file" (or "files").
    # files are concatenated in the given order into a single curve/column.
    [[runs]]
    file = ["slurm-208455.out", "slurm-208634.out"]
    name = "midmax"
"""
import argparse
import os
import re
import sys
import textwrap

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python <= 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

import matplotlib
matplotlib.use("Agg")  # save-to-file only; avoids loading/blocking on a GUI backend
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# strip ANSI escape sequences, then pull step + loss
ANSI = re.compile(r"\x1b\[[0-9;]*m")
VAL = re.compile(r"validate step:\s*(\d+)\s+loss:\s*([\d.]+)")
# training lines look like "step:  2  loss: 12.68598  grad_norm:  1.3970 ..."
# (exclude "validate step:" via a negative lookbehind)
TRAIN = re.compile(r"(?<!validate )step:\s*(\d+)\s+loss:\s*([\d.]+)(?:\s+grad_norm:\s*([\d.]+))?")


def parse(path):
    """Return (train_steps, train_losses, grad_norms, val_steps, val_losses).

    grad_norms is aligned with train_steps; entries are None where a training
    line had no grad_norm field.
    """
    tr_steps, tr_losses, grad_norms, val_steps, val_losses = [], [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = ANSI.sub("", line)
            m = VAL.search(clean)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                continue
            m = TRAIN.search(clean)
            if m:
                tr_steps.append(int(m.group(1)))
                tr_losses.append(float(m.group(2)))
                grad_norms.append(float(m.group(3)) if m.group(3) else None)
    return tr_steps, tr_losses, grad_norms, val_steps, val_losses


def parse_many(paths):
    """Parse several log files and concatenate them (in order) as one run.

    Same return shape as parse(); use this when a single logical run is split
    across multiple .out files (e.g. a resumed job).
    """
    tr_steps, tr_losses, grad_norms, val_steps, val_losses = [], [], [], [], []
    for p in paths:
        a, b, c, d, e = parse(p)
        tr_steps.extend(a)
        tr_losses.extend(b)
        grad_norms.extend(c)
        val_steps.extend(d)
        val_losses.extend(e)
    return tr_steps, tr_losses, grad_norms, val_steps, val_losses


def runs_from_config(path):
    """Return (runs, output, title) from a .toml config.

    runs is a list of (files, label) tuples where files is a list of one or
    more paths (a run may span several .out files). Paths are resolved relative
    to the config file's directory so a config can be run from anywhere.
    """
    if tomllib is None:
        sys.exit("reading a .toml config needs Python 3.11+ or the 'tomli' package (pip install tomli)")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    base = os.path.dirname(os.path.abspath(path))
    runs = []
    for i, r in enumerate(cfg.get("runs", [])):
        # accept "file" (str or list) or "files" (list); a run may span files
        raw = r.get("files", r.get("file"))
        if raw is None:
            sys.exit(f"{path}: runs[{i}] is missing required 'file' key")
        file_list = raw if isinstance(raw, list) else [raw]
        if not file_list:
            sys.exit(f"{path}: runs[{i}] has an empty file list")
        paths = [f if os.path.isabs(f) else os.path.join(base, f) for f in file_list]
        label = r.get("name") or os.path.basename(file_list[0])
        runs.append((paths, label))
    return runs, cfg.get("output"), cfg.get("title")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfiles", nargs="*",
                    help='one or more slurm .out logs; use "path=label" to set a custom legend name')
    ap.add_argument("-c", "--config", help="TOML config file describing runs (see runs.example.toml)")
    ap.add_argument("-o", "--output", help="output image path (default: val_loss.png)")
    ap.add_argument("-t", "--title", help="plot title")
    args = ap.parse_args()

    cfg_out = cfg_title = None
    if args.config:
        if args.logfiles:
            sys.exit("provide runs either positionally or via -c/--config, not both")
        runs, cfg_out, cfg_title = runs_from_config(args.config)
    else:
        if not args.logfiles:
            ap.error("no runs given: pass logfiles positionally or use -c/--config")
        runs = []
        for entry in args.logfiles:
            # allow "path[,path2,...]=custom legend label"; split label on the
            # first "=", then split the path part on "," for multi-file runs
            if "=" in entry:
                path, label = entry.split("=", 1)
            else:
                path, label = entry, None
            paths = path.split(",")
            if label is None:
                label = os.path.basename(paths[0])
            runs.append((paths, label))

    out = args.output or cfg_out or "val_loss.png"
    suptitle = args.title or cfg_title or "GPT-OSS 20b"

    # figure: stacked training-loss (top) and grad-norm (bottom) plots on the
    # left, table spanning both rows on the right
    fig, axd = plt.subplot_mosaic(
        [["loss", "table"],
         ["grad", "table"]],
        figsize=(15, 9),
        width_ratios=[3, 1.4],
        height_ratios=[2, 1],
    )
    ax, ax_grad, ax_tbl = axd["loss"], axd["grad"], axd["table"]

    labels = []                 # column labels, in input order
    loss_by_step = {}           # label -> {val_step: val_loss}  (table data)
    colors = {}                 # label -> line color
    total_val = 0               # validation datapoints (table)

    for paths, label in runs:
        existing = [p for p in paths if os.path.exists(p)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"warning: file(s) not found: {', '.join(missing)}", file=sys.stderr)
        if not existing:
            # keep a placeholder so the run still shows in the legend + table
            (line,) = ax.plot([], [], linewidth=1.2, label=f"{label} (missing)")
            ax_grad.plot([], [], linewidth=1.2, color=line.get_color())
            labels.append(label)
            loss_by_step[label] = {}
            colors[label] = line.get_color()
            continue

        tr_steps, tr_losses, grad_norms, val_steps, val_losses = parse_many(existing)

        # curves plot TRAINING loss (no per-point marker — too dense)
        (line,) = ax.plot(tr_steps, tr_losses, linewidth=1.2, label=label)
        color = line.get_color()

        # grad-norm curve below, in the matching color (skip steps w/o grad_norm)
        gsteps = [s for s, g in zip(tr_steps, grad_norms) if g is not None]
        gvals = [g for g in grad_norms if g is not None]
        ax_grad.plot(gsteps, gvals, linewidth=1.2, color=color, label=label)

        labels.append(label)
        loss_by_step[label] = dict(zip(val_steps, val_losses))
        colors[label] = color
        total_val += len(val_steps)

    if not labels:
        sys.exit("No datapoints found in any input file")

    ax.set_xlabel("step")
    ax.set_ylabel("training loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax_grad.set_xlabel("step")
    ax_grad.set_ylabel("grad norm")
    ax_grad.set_title("Gradient Norm", fontweight="bold")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.sharex(ax)

    # --- table of VALIDATION losses per step for each method ---
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
    ax_tbl.set_title("Validation Loss", fontweight="bold")
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # dark theme: transparent cells with light-gray borders + white text
    for cell in table.get_celld().values():
        cell.set_edgecolor("gray")
        cell.set_facecolor("none")
        cell.set_text_props(color="white")

    # give the header row enough height for the tallest wrapped label
    max_lines = max(lbl.count("\n") + 1 for lbl in col_labels)
    base_h = table[0, 0].get_height()
    # color the header cells to match each method's line
    for c, (raw, lab) in enumerate(zip(["step"] + labels, col_labels)):
        cell = table[0, c]
        cell.set_height(base_h * max_lines)
        cell.set_text_props(weight="bold")
        if raw in colors:
            # method line colors are light pastels on the dark theme,
            # so black header text reads better than white
            cell.set_facecolor(colors[raw])
            cell.set_text_props(weight="bold", color="black")

    # figure-level supertitle above both the plot subtitle and the table title
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))  # leave room for the suptitle
    fig.savefig(out, dpi=150)
    n_files = sum(len(paths) for paths, _ in runs)
    print(f"Wrote {out} ({total_val} validation datapoints in table across "
          f"{len(runs)} run(s), {n_files} file(s))")


if __name__ == "__main__":
    main()
