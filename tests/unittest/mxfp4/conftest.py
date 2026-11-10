# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

import sys

from tabulate import tabulate


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    del exitstatus, config
    for module in sys.modules.values():
        results = getattr(module, "_RESULTS", None)
        headers = getattr(module, "_SUMMARY_HEADERS", None)
        path = getattr(module, "__file__", None) or ""
        if not results or not headers or not path.endswith("test_mxfp_attention.py"):
            continue
        terminalreporter.write_sep("-", "MXFP4 attention vs bf16 SDPA")
        terminalreporter.write_line(tabulate(results, headers=headers, tablefmt="github", floatfmt=".4f"))
        return
