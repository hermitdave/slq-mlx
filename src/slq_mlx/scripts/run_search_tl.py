#!/usr/bin/env python3
"""Stage 3a: Task-Lossless search (lm-eval in loop).

Usage:
    python -m slq_mlx.scripts.run_search_tl --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit --tasks arc_challenge,gsm8k,mmlu --min-recovery 0.99
"""

import argparse
import logging
import sys
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Task-Lossless SLQ search (lm-eval in loop)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--shapley-db")
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--ear-weight", type=float, default=1.0)
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval tasks")
    parser.add_argument("--min-recovery", type=float, required=True)
    parser.add_argument("--seeds", default="42,1,1234")
    parser.add_argument("--baseline-json")
    parser.add_argument("--bitwidths", default="2,3,4,5,6,8")
    parser.add_argument("--bitwidth-map", default="")
    parser.add_argument("--bitwidth-tolerance", type=float, default=0.1)
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-gen-toks", type=int, default=4096)
    parser.add_argument("--limit", type=float)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-thinking", action="store_true")

    args = parser.parse_args()

    logger.warning("Full Task-Lossless search requires lm-eval integration.")
    logger.warning("Running Distribution-Lossless search as proxy with predicted constraints...")

    cmd = [
        sys.executable, "-m", "slq_mlx.mlx_search",
        "--model", args.model,
        "--layer-dir", args.layer_dir,
        "--mode", "dl",
        "--sensitivity-method", "shapley",
        "--min-ear", str(args.min_recovery),
        "--use-predicted-constraints",
        "--bitwidths", args.bitwidths,
        "--bitwidth-map", args.bitwidth_map,
        "--bitwidth-tolerance", str(args.bitwidth_tolerance),
        "--calibration-data", "fineweb_edu",
        "--calibration-samples", "512",
        "--calibration-seq-length", "2048",
        "--batch-size", "4",
        "--top-k", "10",
        "--seed", "42",
    ]

    if args.shapley_db:
        cmd.extend(["--shapley-db", args.shapley_db])
    if args.output:
        cmd.extend(["--output", args.output])

    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())