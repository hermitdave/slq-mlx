#!/usr/bin/env python3
"""Stage 3b: Distribution-Lossless search (EAR target).

Usage:
    python -m slq_mlx.scripts.run_search_dl --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit --min-ear 0.99
"""

import argparse
import logging
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Distribution-Lossless SLQ search")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--mode", choices=["dl", "tl"], default="dl")
    parser.add_argument("--sensitivity-method", choices=["shapley", "linear"], default="shapley")
    parser.add_argument("--shapley-db")
    parser.add_argument("--linear-db")
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--ear-weight", type=float)
    parser.add_argument("--min-ear", type=float)
    parser.add_argument("--target-bitwidth", type=float)
    parser.add_argument("--use-predicted-constraints", action="store_true")
    parser.add_argument("--bitwidths")
    parser.add_argument("--bitwidth-map", default="")
    parser.add_argument("--bitwidth-tolerance", type=float, default=0.1)
    parser.add_argument("--disable-grouping", action="store_true")
    parser.add_argument("--calibration-data", default="fineweb_edu")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")

    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "slq_mlx.mlx_search",
        "--model", args.model,
        "--layer-dir", args.layer_dir,
        "--mode", args.mode,
        "--sensitivity-method", args.sensitivity_method,
    ]

    if args.shapley_db:
        cmd.extend(["--shapley-db", args.shapley_db])
    if args.linear_db:
        cmd.extend(["--linear-db", args.linear_db])
    if args.kl_weight is not None:
        cmd.extend(["--kl-weight", str(args.kl_weight)])
    if args.ear_weight is not None:
        cmd.extend(["--ear-weight", str(args.ear_weight)])
    if args.min_ear is not None:
        cmd.extend(["--min-ear", str(args.min_ear)])
    if args.target_bitwidth is not None:
        cmd.extend(["--target-bitwidth", str(args.target_bitwidth)])
    if args.use_predicted_constraints:
        cmd.append("--use-predicted-constraints")
    if args.bitwidths:
        cmd.extend(["--bitwidths", args.bitwidths])
    if args.bitwidth_map:
        cmd.extend(["--bitwidth-map", args.bitwidth_map])
    if args.bitwidth_tolerance:
        cmd.extend(["--bitwidth-tolerance", str(args.bitwidth_tolerance)])
    if args.disable_grouping:
        cmd.append("--disable-grouping")
    cmd.extend([
        "--calibration-data", args.calibration_data,
        "--calibration-samples", str(args.calibration_samples),
        "--calibration-seq-length", str(args.calibration_seq_length),
        "--batch-size", str(args.batch_size),
        "--top-k", str(args.top_k),
        "--seed", str(args.seed),
    ])
    if args.output:
        cmd.extend(["--output", args.output])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())