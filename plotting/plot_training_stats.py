#!/usr/bin/env python3
"""Plot training/validation loss and grad norm from slurm .out logs and/or
TensorBoard event files.

Two source kinds are supported and may be freely mixed within a single plot
(and even within a single run):

  * slurm .out logs -- parsed by regex from lines like:
        ... validate step: 768  loss:  4.9572  memory: ...
    (ANSI color codes are stripped before matching.)

  * TensorBoard sources -- either an event file (events.out.tfevents.*) or a
    directory containing them (e.g. a run's tb/ dir, whose timestamped subdirs
    from resumes are read in order). Scalar tags read:
        loss_metrics/global_avg_loss  -> training loss
        grad_norm                     -> gradient norm
        validation_metrics/loss       -> validation loss

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

A fourth subplot (exponent/mantissa update RMS ratio) is drawn automatically
when any run's log contains m_adam diagnostic lines:
    [m_adam] step: 42  rms_dw_exp: 1.23e-05  rms_dw_man: 9.87e-06  exp/man: 1.25
Force it on/off with --madam-rms / --no-madam-rms (or show_madam_rms in a config).

TOML config format (see runs.example.toml):
    output   = "val_loss.png"               # optional, overridden by -o
    title    = "Validation loss vs. step"   # optional
    max_step = 5000                         # optional, global step cutoff (-m)
    show_madam_rms = true                   # optional: force the m_adam
                                            # exp/mantissa ratio subplot on/off
                                            # (default: auto-detect from logs)

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

    # a TensorBoard run: point "file" at the tb dir (or a single event file).
    # .out logs and tb sources may be mixed in the same list and across runs.
    [[runs]]
    file = "gpt_oss_20b-pretrain-bf16/tb"
    name = "bf16 (tb)"

    # limit how far along the x-axis data is shown. a top-level `max_step`
    # applies to every run; a per-run `max_step` overrides it for that run.
    # (also settable on the CLI with --max-step.)
    max_step = 5000                             # optional, global cutoff

    [[runs]]
    file = "slurm-209000.out"
    name = "short view"
    max_step = 2000                             # optional, per-run cutoff
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
# m_adam optimizer diagnostic lines look like:
#   "[m_adam] step: 42  rms_dw_exp: 1.234567e-05  rms_dw_man: 9.876543e-06  exp/man: 1.2500"
# (emitted by MAdamOptimizersContainer when track_dw_rms is on)
MADAM = re.compile(
    r"\[m_adam\]\s+step:\s*(\d+)\s+rms_dw_exp:\s*([\d.eE+-]+)\s+rms_dw_man:\s*([\d.eE+-]+)"
)


# TensorBoard scalar tags: training loss, grad norm, validation loss.
# Matched exactly, then by suffix, so a loose name (e.g. "global_avg_loss")
# still finds the full tag "loss_metrics/global_avg_loss".
TB_TRAIN_TAG = "loss_metrics/global_avg_loss"
TB_GRAD_TAG = "grad_norm"
TB_VAL_TAG = "validation_metrics/loss"
# m_adam exponent/mantissa weight-change RMS (only present if wired to TB).
TB_MADAM_EXP_TAG = "rms_dw_exp"
TB_MADAM_MAN_TAG = "rms_dw_man"


def is_tb(path):
    """True if path looks like a TensorBoard source (event file or dir of them)."""
    if os.path.isdir(path):
        return True
    return os.path.basename(path).startswith("events.out.tfevents")


def _tb_event_files(path):
    """Return event files for a TB source, sorted so resumes concatenate in order.

    A path may be a single event file, or a directory holding one or more of
    them (e.g. a run's tb/ dir with a timestamped subdir per resume).
    """
    if os.path.isdir(path):
        import glob
        return sorted(glob.glob(os.path.join(path, "**", "events.out.tfevents.*"),
                                recursive=True))
    return [path]


def _tb_scalars(ea, tag):
    """Return [(step, value), ...] for tag, matching exactly or by suffix.

    Returns [] if no tag matches (e.g. an event file with no validation data).
    """
    tags = ea.Tags().get("scalars", [])
    if tag not in tags:
        matches = [t for t in tags if t.endswith("/" + tag) or t == tag]
        if not matches:
            return []
        tag = matches[0]
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def parse_tb(path):
    """Parse TensorBoard scalars; same return shape as parse().

    Reads global_avg_loss (training loss), grad_norm, and
    validation_metrics/loss (validation loss). grad_norms is aligned with
    train_steps (None where a step has no grad_norm). Multiple event files
    under a directory are read in order and concatenated as one run.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError:
        sys.exit("reading TensorBoard logs needs the 'tensorboard' package "
                 "(pip install tensorboard)")

    tr_steps, tr_losses, grad_norms, val_steps, val_losses = [], [], [], [], []
    md_steps, md_exp, md_man = [], [], []
    for ef in _tb_event_files(path):
        # size_guidance scalars=0 disables TB's default downsampling to 1000 pts
        ea = EventAccumulator(ef, size_guidance={"scalars": 0})
        ea.Reload()
        grad = dict(_tb_scalars(ea, TB_GRAD_TAG))
        for step, loss in _tb_scalars(ea, TB_TRAIN_TAG):
            tr_steps.append(step)
            tr_losses.append(loss)
            grad_norms.append(grad.get(step))
        for step, loss in _tb_scalars(ea, TB_VAL_TAG):
            val_steps.append(step)
            val_losses.append(loss)
        # m_adam RMS (usually absent from TB — only if explicitly wired there)
        exp = dict(_tb_scalars(ea, TB_MADAM_EXP_TAG))
        man = dict(_tb_scalars(ea, TB_MADAM_MAN_TAG))
        for step in sorted(set(exp) & set(man)):
            md_steps.append(step)
            md_exp.append(exp[step])
            md_man.append(man[step])
    return (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
            md_steps, md_exp, md_man)


def parse(path):
    """Return (train_steps, train_losses, grad_norms, val_steps, val_losses,
    madam_steps, madam_rms_exp, madam_rms_man).

    grad_norms is aligned with train_steps; entries are None where a training
    line had no grad_norm field. The madam_* arrays are aligned with each other
    (one entry per '[m_adam]' diagnostic line) and are empty for non-m_adam runs.
    """
    tr_steps, tr_losses, grad_norms, val_steps, val_losses = [], [], [], [], []
    md_steps, md_exp, md_man = [], [], []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = ANSI.sub("", line)
            m = VAL.search(clean)
            if m:
                val_steps.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                continue
            m = MADAM.search(clean)
            if m:
                md_steps.append(int(m.group(1)))
                md_exp.append(float(m.group(2)))
                md_man.append(float(m.group(3)))
                continue
            m = TRAIN.search(clean)
            if m:
                tr_steps.append(int(m.group(1)))
                tr_losses.append(float(m.group(2)))
                grad_norms.append(float(m.group(3)) if m.group(3) else None)
    return (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
            md_steps, md_exp, md_man)


def parse_many(paths):
    """Parse several sources and concatenate them (in order) as one run.

    Same return shape as parse(); use this when a single logical run is split
    across multiple sources (e.g. a resumed job). Each source may be a slurm
    .out log or a TensorBoard event file/dir; the two may be freely mixed.
    """
    tr_steps, tr_losses, grad_norms, val_steps, val_losses = [], [], [], [], []
    md_steps, md_exp, md_man = [], [], []
    for p in paths:
        a, b, c, d, e, f, g, h = parse_tb(p) if is_tb(p) else parse(p)
        tr_steps.extend(a)
        tr_losses.extend(b)
        grad_norms.extend(c)
        val_steps.extend(d)
        val_losses.extend(e)
        md_steps.extend(f)
        md_exp.extend(g)
        md_man.extend(h)
    # a resumed job replays earlier steps; keep only the latest datapoint per
    # step (last occurrence in concat order wins), then sort ascending by step
    tr_steps, (tr_losses, grad_norms) = _dedup_latest(tr_steps, tr_losses, grad_norms)
    val_steps, (val_losses,) = _dedup_latest(val_steps, val_losses)
    md_steps, (md_exp, md_man) = _dedup_latest(md_steps, md_exp, md_man)
    return (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
            md_steps, md_exp, md_man)


def _dedup_latest(steps, *aligned):
    """Collapse duplicate steps, keeping the last occurrence, sorted by step.

    `aligned` are lists parallel to `steps`. Returns (sorted_steps, (col, ...))
    where each col is the corresponding aligned list, filtered and reordered to
    match. Iterating in order and writing into a dict makes later (resumed-run)
    datapoints overwrite earlier ones at the same step.
    """
    latest = {}
    for i, s in enumerate(steps):
        latest[s] = tuple(col[i] for col in aligned)
    ordered = sorted(latest)
    cols = tuple([latest[s][j] for s in ordered] for j in range(len(aligned)))
    return ordered, cols


def clip_to_step(data, max_step):
    """Trim parsed data to steps <= max_step (no-op if max_step is None).

    `data` is the 5-tuple returned by parse()/parse_many(); training arrays
    (steps/losses/grad_norms) stay aligned, as do the validation arrays.
    """
    (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
     md_steps, md_exp, md_man) = data
    if max_step is None:
        return data
    tr = [(s, l, g) for s, l, g in zip(tr_steps, tr_losses, grad_norms) if s <= max_step]
    va = [(s, l) for s, l in zip(val_steps, val_losses) if s <= max_step]
    md = [(s, e, mn) for s, e, mn in zip(md_steps, md_exp, md_man) if s <= max_step]
    tr_steps, tr_losses, grad_norms = map(list, zip(*tr)) if tr else ([], [], [])
    val_steps, val_losses = map(list, zip(*va)) if va else ([], [])
    md_steps, md_exp, md_man = map(list, zip(*md)) if md else ([], [], [])
    return (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
            md_steps, md_exp, md_man)


def runs_from_config(path):
    """Return (runs, output, title, details, max_step, show_madam_rms) from a
    .toml config.

    runs is a list of (files, label, max_step) tuples where files is a list of
    one or more paths (a run may span several .out files) and max_step is an
    optional per-run step cutoff (None if unset). Paths are resolved relative
    to the config file's directory so a config can be run from anywhere. The
    returned top-level max_step is the global cutoff (None if unset).
    show_madam_rms is the optional top-level flag toggling the m_adam
    exponent/mantissa RMS-ratio subplot (None if unset -> auto-detect).
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
        runs.append((paths, label, r.get("max_step")))
    return (runs, cfg.get("output"), cfg.get("title"), cfg.get("details"),
            cfg.get("max_step"), cfg.get("show_madam_rms"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfiles", nargs="*",
                    help='one or more slurm .out logs or TensorBoard sources '
                         '(event file or tb/ dir); use "path=label" to set a '
                         'custom legend name')
    ap.add_argument("-c", "--config", help="TOML config file describing runs (see runs.example.toml)")
    ap.add_argument("-o", "--output", help="output image path (default: val_loss.png)")
    ap.add_argument("-t", "--title", help="plot title")
    ap.add_argument("-d", "--details", help="extra info shown in a small text box (top-right)")
    ap.add_argument("-m", "--max-step", type=int,
                    help="only show data up to this step (applies to all runs; "
                         "overrides per-run/global max_step in a config)")
    ap.add_argument("--madam-rms", dest="madam_rms", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="show (or hide with --no-madam-rms) the m_adam "
                         "exponent/mantissa update RMS-ratio subplot; default "
                         "auto-detects from the data")
    args = ap.parse_args()

    cfg_out = cfg_title = cfg_details = cfg_max_step = cfg_show_madam = None
    if args.config:
        if args.logfiles:
            sys.exit("provide runs either positionally or via -c/--config, not both")
        (runs, cfg_out, cfg_title, cfg_details, cfg_max_step,
         cfg_show_madam) = runs_from_config(args.config)
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
            runs.append((paths, label, None))

    out = args.output or cfg_out or "val_loss.png"
    suptitle = args.title or cfg_title or "GPT-OSS 20b"
    details = args.details or cfg_details
    # global step cutoff: CLI wins, else config's top-level max_step. A per-run
    # max_step (set only via config) overrides this for that run.
    global_max_step = args.max_step if args.max_step is not None else cfg_max_step
    # m_adam RMS subplot: CLI wins, else config's show_madam_rms, else auto.
    show_madam = args.madam_rms if args.madam_rms is not None else cfg_show_madam

    # --- pass 1: resolve + parse every run up front, so the layout (whether to
    # draw the m_adam subplot) can be decided before the figure is built ---
    parsed = []  # list of {label, data} where data is a parsed tuple or None
    for paths, label, run_max_step in runs:
        # per-run max_step overrides the global cutoff; the global applies otherwise
        max_step = run_max_step if run_max_step is not None else global_max_step
        existing = [p for p in paths if os.path.exists(p)]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            print(f"warning: file(s) not found: {', '.join(missing)}", file=sys.stderr)
        data = clip_to_step(parse_many(existing), max_step) if existing else None
        parsed.append({"label": label, "data": data})

    # auto-detect: draw the ratio subplot when any run has m_adam RMS data,
    # unless the flag (CLI/config) forces it on or off.
    has_madam = any(d["data"] and d["data"][5] for d in parsed)
    draw_madam = has_madam if show_madam is None else bool(show_madam)

    # figure: stacked training-loss (top), grad-norm, optional exponent/mantissa
    # RMS-ratio, and validation-loss plots on the left, table on the right. When
    # `details` is given, a small text box takes a thin strip at the top of the
    # right column (beside training loss, above the table + its title); the table
    # keeps most of the column height so it stays uncramped.
    n_left = 4 if draw_madam else 3
    height_ratios = [2, 1, 1, 1] if draw_madam else [2, 1, 1]
    fig = plt.figure(figsize=(15, 13 if draw_madam else 11))
    gs = fig.add_gridspec(n_left, 2, width_ratios=[3, 1.4], height_ratios=height_ratios)
    ax = fig.add_subplot(gs[0, 0])
    ax_grad = fig.add_subplot(gs[1, 0])
    ax_madam = fig.add_subplot(gs[2, 0]) if draw_madam else None
    ax_val = fig.add_subplot(gs[n_left - 1, 0])
    ax_det = None
    if details:
        gs_r = gs[:, 1].subgridspec(2, 1, height_ratios=[1, 5])
        ax_det = fig.add_subplot(gs_r[0])
        ax_tbl = fig.add_subplot(gs_r[1])
    else:
        ax_tbl = fig.add_subplot(gs[:, 1])

    labels = []                 # column labels, in input order
    loss_by_step = {}           # label -> {val_step: val_loss}  (table data)
    colors = {}                 # label -> line color
    total_val = 0               # validation datapoints (table)
    madam_has_pos = False       # any positive ratio plotted (-> log y-scale)

    # --- pass 2: draw each run ---
    for entry in parsed:
        label = entry["label"]
        data = entry["data"]
        if data is None:
            # keep a placeholder so the run still shows in the legend + table
            (line,) = ax.plot([], [], linewidth=1.2, label=f"{label} (missing)")
            ax_grad.plot([], [], linewidth=1.2, color=line.get_color())
            if ax_madam is not None:
                ax_madam.plot([], [], linewidth=1.2, color=line.get_color())
            ax_val.plot([], [], linewidth=1.2, color=line.get_color())
            labels.append(label)
            loss_by_step[label] = {}
            colors[label] = line.get_color()
            continue

        (tr_steps, tr_losses, grad_norms, val_steps, val_losses,
         md_steps, md_exp, md_man) = data

        # curves plot TRAINING loss (no per-point marker — too dense)
        (line,) = ax.plot(tr_steps, tr_losses, linewidth=1.2, label=label)
        color = line.get_color()

        # grad-norm curve below, in the matching color (skip steps w/o grad_norm)
        gsteps = [s for s, g in zip(tr_steps, grad_norms) if g is not None]
        gvals = [g for g in grad_norms if g is not None]
        ax_grad.plot(gsteps, gvals, linewidth=1.2, color=color, label=label)

        # exponent/mantissa update RMS ratio: >1 means the exponent branch moves
        # weights harder than the AdamW mantissa branch (skip steps w/o a
        # positive denominator, e.g. lr_e=0 runs where rms_dw_man could be 0)
        if ax_madam is not None:
            rsteps = [s for s, m in zip(md_steps, md_man) if m > 0]
            ratio = [e / m for e, m in zip(md_exp, md_man) if m > 0]
            ax_madam.plot(rsteps, ratio, linewidth=1.2, color=color, label=label)
            if any(r > 0 for r in ratio):
                madam_has_pos = True

        # validation-loss curve, matching color (markers since points are sparse)
        ax_val.plot(val_steps, val_losses, linewidth=1.2, marker="o",
                    markersize=3, color=color, label=label)

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
    if global_max_step is not None:
        ax.set_xlim(right=global_max_step)  # grad/madam/val share this x-axis

    ax_grad.set_xlabel("step")
    ax_grad.set_ylabel("grad norm")
    ax_grad.set_title("Gradient Norm", fontweight="bold")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.sharex(ax)

    if ax_madam is not None:
        # y=1 reference: exponent branch moving weights exactly as hard as the
        # mantissa (AdamW) branch. log scale so 0.1x .. 10x reads symmetrically.
        ax_madam.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=1.0)
        if madam_has_pos:
            ax_madam.set_yscale("log")
        ax_madam.set_xlabel("step")
        ax_madam.set_ylabel("exp / mantissa")
        ax_madam.set_title("Exponent/Mantissa Update RMS Ratio", fontweight="bold")
        ax_madam.grid(True, alpha=0.3)
        ax_madam.sharex(ax)

    ax_val.set_xlabel("step")
    ax_val.set_ylabel("validation loss")
    ax_val.set_title("Validation Loss", fontweight="bold")
    ax_val.grid(True, alpha=0.3)
    ax_val.sharex(ax)

    # --- optional details text box (top-right, above the table) ---
    if ax_det is not None:
        ax_det.axis("off")
        ax_det.text(
            0.5, 0.98, textwrap.fill(details, width=34),
            transform=ax_det.transAxes, ha="center", va="top",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="none", edgecolor="gray"),
        )

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
    # also emit vector PDF + SVG alongside the raster output
    base = os.path.splitext(out)[0]
    pdf_out, svg_out = base + ".pdf", base + ".svg"
    fig.savefig(pdf_out)
    fig.savefig(svg_out)
    n_files = sum(len(paths) for paths, *_ in runs)
    print(f"Wrote {out}, {pdf_out} and {svg_out} ({total_val} validation "
          f"datapoints in table across {len(runs)} run(s), {n_files} file(s))")


if __name__ == "__main__":
    main()
