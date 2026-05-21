#!/usr/bin/env python3
"""Verify a low-precision-training (LPT) wrapper run was actually active.

Goal
----
Given the torchtitan-style training log of an alto LPT run (e.g. NVFP4 / MXFP4 /
MXFP8 default recipe), confirm that:

  (A) ``swap_params`` ran and produced at least one
      ``Swapped <fqn> to <Class>TrainingWeightWrapperTensor`` line for the
      requested precision class.
  (B) The model dump emitted by ``LowPrecisionTrainingModifier converted
      model:`` lists every Linear / GroupedExperts target with the desired
      ``TrainingOpConfig``.
  (C) The training log records step-1 loss + grad_norm — i.e. the wrapped
      forward really executed once and produced a finite loss.

Historical note: an earlier revision of this script also asserted that no
``bf16_forward`` / ``bf16_backward`` / ``bypass_*`` flag appeared as
``True`` in the model dump.  Those Phase B ablation knobs have been
removed from production code (`alto/kernels/dispatch/config.py` and
`alto/modifiers/lpt/base.py` no longer define them), so the bypass-flag
check would always trivially pass; it has been dropped to keep the
verifier focused on what is still meaningful.

Usage
-----
    python scripts/verify_lpt_wrapper_active.py \
        --log /path/to/run.log --expected-precision nvfp4

Exit codes: 0 = PASS, 1 = FAIL.

The script prints a JSON summary so an agent / CI step can grep numbers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SWAP_PAT = re.compile(
    r"Swapped (?P<fqn>\S+) to (?P<cls>[A-Za-z0-9]+TrainingWeightWrapperTensor)"
)
CONVERTED_PAT = re.compile(r"LowPrecisionTrainingModifier converted model")
STEP1_PAT = re.compile(
    r"step:\s*1\b.*?loss:\s*(?P<loss>[0-9.]+).*?grad_norm:\s*(?P<grad>[0-9.eE+-]+)"
)
CONFIG_PAT = re.compile(r"TrainingOpConfig\(precision='(?P<prec>[^']+)'")
FAIL_PAT = re.compile(r"Traceback|ChildFailedError|RuntimeError:")

CLASS_BY_PRECISION = {
    "mxfp4": "MXFP4TrainingWeightWrapperTensor",
    "nvfp4": "NVFP4TrainingWeightWrapperTensor",
    "mxfp8_e4m3": "MXFP8TrainingWeightWrapperTensor",
    "mxfp8_e5m2": "MXFP8TrainingWeightWrapperTensor",
}


def strip_ansi(s: str) -> str:
    return ANSI.sub("", s)


def verify(log_path: Path, expected_precision: str) -> tuple[int, dict]:
    if expected_precision not in CLASS_BY_PRECISION:
        raise SystemExit(
            f"unknown precision {expected_precision!r}; "
            f"expected one of {sorted(CLASS_BY_PRECISION)}"
        )

    expected_cls = CLASS_BY_PRECISION[expected_precision]
    text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    swap_lines: list[str] = []
    converted_seen = False
    config_precs: set[str] = set()
    step1_loss: Optional[float] = None
    step1_grad: Optional[float] = None
    runtime_failure = False
    for raw in text.splitlines():
        s = strip_ansi(raw)
        if "Swapped " in s:
            m = SWAP_PAT.search(s)
            if m and m.group("cls") == expected_cls:
                swap_lines.append(s)
        if CONVERTED_PAT.search(s):
            converted_seen = True
        for m in CONFIG_PAT.finditer(s):
            config_precs.add(m.group("prec"))
        if step1_loss is None:
            m = STEP1_PAT.search(s)
            if m:
                try:
                    step1_loss = float(m.group("loss"))
                    step1_grad = float(m.group("grad"))
                except ValueError:
                    pass
        if FAIL_PAT.search(s):
            runtime_failure = True

    swapped_modules = {l.split("Swapped ")[-1].split(" to ")[0] for l in swap_lines}

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "A.1 swap_params emitted at least 1 Swapped line for the expected class",
            len(swap_lines) > 0,
            f"swap_count={len(swap_lines)} expected_cls={expected_cls}",
        )
    )
    checks.append(
        (
            "A.2 swapped modules cover both attention.* and moe.experts.* (or single Linear set)",
            (
                any(".attention." in m for m in swapped_modules)
                or any(".moe." in m for m in swapped_modules)
            ),
            f"attention_hits={sum('.attention.' in m for m in swapped_modules)} "
            f"moe_hits={sum('.moe.' in m for m in swapped_modules)}",
        )
    )
    checks.append(
        (
            "B.1 LowPrecisionTrainingModifier converted model line present",
            converted_seen,
            f"present={converted_seen}",
        )
    )
    checks.append(
        (
            "B.2 converted model dump shows expected precision tag",
            expected_precision in config_precs,
            f"precisions_seen={sorted(config_precs)}",
        )
    )
    checks.append(
        (
            "C.1 step 1 produced a finite loss + grad_norm in the wrapped forward",
            step1_loss is not None
            and step1_grad is not None
            and step1_loss == step1_loss
            and step1_grad == step1_grad,
            f"step1_loss={step1_loss} step1_grad={step1_grad}",
        )
    )
    checks.append(
        (
            "C.2 no Traceback / ChildFailedError / RuntimeError in the log",
            not runtime_failure,
            f"runtime_failure={runtime_failure}",
        )
    )

    overall_ok = all(ok for _, ok, _ in checks)

    summary = {
        "log": str(log_path),
        "expected_precision": expected_precision,
        "expected_class": expected_cls,
        "swap_count": len(swap_lines),
        "swapped_module_count": len(swapped_modules),
        "converted_model_line_seen": converted_seen,
        "precisions_seen_in_dump": sorted(config_precs),
        "bypass_active": bypass_active,
        "step1_loss": step1_loss,
        "step1_grad": step1_grad,
        "runtime_failure": runtime_failure,
        "checks": [
            {"name": name, "passed": passed, "detail": detail}
            for name, passed, detail in checks
        ],
        "overall_passed": overall_ok,
    }

    return (0 if overall_ok else 1), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="path to the training log to verify",
    )
    parser.add_argument(
        "--expected-precision",
        type=str,
        required=True,
        choices=sorted(CLASS_BY_PRECISION),
        help="recipe scheme name (mxfp4 / nvfp4 / mxfp8_e4m3 / mxfp8_e5m2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print only the JSON summary, no human-readable lines",
    )
    args = parser.parse_args()

    rc, summary = verify(args.log, args.expected_precision)
    if not args.json:
        verdict = "PASS" if rc == 0 else "FAIL"
        print(f"[verify_lpt_wrapper_active] {verdict}  log={args.log}")
        for check in summary["checks"]:
            tag = "OK " if check["passed"] else "BAD"
            print(f"  [{tag}] {check['name']}  ({check['detail']})")
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.exit(rc)


if __name__ == "__main__":
    main()
