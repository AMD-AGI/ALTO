# Modifications Copyright (c) Advanced Micro Devices, Inc.
# All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import shutil
from pathlib import Path

from torchtitan.config.manager import ConfigManager
from torchtitan.config.job_config import JobConfig

from modeloptimizer.utils.exportation import convert_to_hf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DCP weights to HF format.")
    parser.add_argument("config_file",
                        type=Path,
                        help="Path to job config file.")
    parser.add_argument(
        "--export_dtype",
        type=str,
        nargs="?",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="Export dtype for HF checkpoint (default: float32)",
    )
    parser.add_argument(
        "--disable_sparse_compression",
        action="store_true",
        help="Disable sparse compression (default: False)",
    )
    parser.add_argument(
        "--save_uncompressed",
        action="store_true",
        help="Save uncompressed checkpoint (default: False)",
    )
    args = parser.parse_args()

    config_manager = ConfigManager()
    job_config: JobConfig = config_manager.parse_args(
        ["--job.config_file", args.config_file.as_posix()])

    dump_folder = Path(job_config.job.dump_folder)
    hf_assets_path = Path(job_config.model.hf_assets_path)
    input_dir = dump_folder / "checkpoint" / f"step-{job_config.training.steps}"
    output_dir = dump_folder / "hf"

    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ["*.json", "*.py"]:
        for extra_file in hf_assets_path.glob(pattern):
            shutil.copy(extra_file, output_dir)

    convert_to_hf(
        job_config,
        input_dir.as_posix(),
        output_dir.as_posix(),
        job_config.model.name,
        job_config.model.flavor,
        hf_assets_path.as_posix(),
        args.export_dtype,
        save_compressed=not args.save_uncompressed,
        disable_sparse_compression=args.disable_sparse_compression,
    )
    sharded_output_dir = output_dir / "sharded"
    shutil.rmtree(sharded_output_dir, ignore_errors=True)
