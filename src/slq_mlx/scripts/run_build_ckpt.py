#!/usr/bin/env python3
"""Export an SLQ store as a ready-to-serve checkpoint.

Default: a packed MLX-native checkpoint (``run_pack_mlx``). The decoder Linear
layers become fused ``nn.QuantizedLinear`` layers, so the model honours the SLQ
bitwidth in memory and on disk -- on Qwen3-1.7B, 1.40 GiB against a 3.21 GiB
BF16 base (2.29x smaller).

Pass ``--dequantized`` for the older fake-quantized float16 artifact instead.
That one is always BF16-sized regardless of the SLQ bitwidth, so it exists only
for debugging or for runtimes without a quantized GEMM.

Usage:
    # packed (default)
    python -m slq_mlx.scripts.run_build_ckpt \\
        --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit \\
        --config ./quant/Qwen3-8B/4bit/config_..._bw4.49_seed42.txt \\
        --output-dir ~/Projects/ml/models

    # fake-quantized float16 (debug only)
    python -m slq_mlx.scripts.run_build_ckpt ... --dequantized
"""

import argparse
import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export an SLQ store as a checkpoint")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--layer-dir", required=True, help="quantized layer store")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="per-layer bitwidth config from the search")
    source.add_argument("--uniform", type=int, metavar="BITS",
                        help="every layer at BITS (calibration anchor)")
    parser.add_argument("--output-dir", required=True,
                        help="writes <output-dir>/<model>/<subdir>")
    parser.add_argument("--subdir-name")
    parser.add_argument("--dequantized", action="store_true",
                        help="write fake-quantized float16 instead of the packed "
                             "checkpoint (always BF16-sized -- debug/fallback only)")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.dequantized:
        script = "slq_mlx.scripts.run_build_ckpt_dequantized"
    else:
        script = "slq_mlx.scripts.run_pack_mlx"

    cmd = [
        sys.executable, "-m", script,
        "--model", args.model,
        "--layer-dir", args.layer_dir,
        "--output-dir", args.output_dir,
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
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
