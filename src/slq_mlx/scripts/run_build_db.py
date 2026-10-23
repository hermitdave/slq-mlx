#!/usr/bin/env python3
"""Stage 2: Build sensitivity database (linear alphas + Shapley values).

Usage:
    python -m slq_mlx.scripts.run_build_db --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit
"""

import argparse
import logging
import sys

from slq_mlx.mlx_sensitivity import run_build_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build SLQ sensitivity databases")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--calibration-data", default="fineweb_edu", choices=["c4", "wikitext2", "fineweb_edu", "gsm8k", "openthoughts"])
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-noise-samples", type=int, default=15)
    parser.add_argument("--skip-linear", action="store_true")
    parser.add_argument("--shapley-bitwidths", default="2,3,4,5,6,7,8")
    parser.add_argument("--num-permutations", type=int, default=50)
    parser.add_argument("--skip-shapley", action="store_true")
    parser.add_argument("--layer-groups", default="",
                        help="Groups of layers constrained to share a bitwidth (space-separated groups, "
                             "comma-separated patterns). Default empty = per-layer, which is correct for "
                             "MLX: MLX does not fuse SwiGLU or QKV, so the upstream grouping only exists "
                             "to keep vLLM's fused projections at a shared bitwidth. Pass the upstream "
                             "spec to reproduce their numbers: "
                             "'mlp.gate_proj,mlp.up_proj,mlp.down_proj "
                             "self_attn.k_proj,self_attn.q_proj,self_attn.v_proj,self_attn.o_proj'")
    parser.add_argument("--allow-per-layer-sensitivity", action="store_true",
                        help="Estimate linear/Shapley sensitivity at per-layer granularity. Expensive: "
                             "cost scales with the number of groups, so 196 layers is ~9 h (linear) and "
                             "~172 h (Shapley) at 10.5 s per metric call. Only meaningful for small models.")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()

    if not args.layer_groups.strip() and not args.allow_per_layer_sensitivity:
        if not args.skip_linear or not args.skip_shapley:
            parser.error(
                "per-layer sensitivity estimation is impractical at this scale "
                "(~9 h linear, ~172 h Shapley for 196 layers). The heuristic "
                "allocator (run_search_dl.py --method heuristic) needs only "
                "error_db.json and is the default. Pass "
                "--allow-per-layer-sensitivity to override, or set --layer-groups "
                "to a coarse grouping."
            )

    linear_path, shapley_path = run_build_database(
        model_id=args.model,
        layer_dir=args.layer_dir,
        calibration_data=args.calibration_data,
        calibration_samples=args.calibration_samples,
        calibration_seq_len=args.calibration_seq_length,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        num_noise_samples=args.num_noise_samples,
        skip_linear=args.skip_linear,
        shapley_bitwidths=args.shapley_bitwidths,
        num_permutations=args.num_permutations,
        layer_groups=args.layer_groups,
        force_recompute=args.force_recompute,
    )

    logger.info(f"Linear DB: {linear_path}")
    logger.info(f"Shapley DB: {shapley_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())