#!/usr/bin/env python3
"""Stage 4: Build MLX checkpoint from SLQ config.

Usage:
    python -m slq_mlx.scripts.run_build_ckpt --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit --config ./quant/Qwen3-8B/4bit/config_dl_ear0.99.txt --output-dir ~/Projects/ml/models
"""

import argparse
import logging
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build MLX checkpoint from SLQ bitwidth config")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--config", help="per-layer bitwidth config (layer: bits) from search")
    parser.add_argument("--uniform", type=int, metavar="BITS", help="every layer at BITS")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subdir-name")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "slq_mlx.mlx_checkpoint",
        "--model", args.model,
        "--layer-dir", args.layer_dir,
        "--output-dir", args.output_dir,
        "--dtype", args.dtype,
        "--group-size", str(args.group_size),
    ]

    if args.config:
        cmd.extend(["--config", args.config])
    if args.uniform is not None:
        cmd.extend(["--uniform", str(args.uniform)])
    if args.subdir_name:
        cmd.extend(["--subdir-name", args.subdir_name])
    if args.force:
        cmd.append("--force")

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())