#!/usr/bin/env python3
"""Stage 2: Build sensitivity database (linear alphas + Shapley values).

Usage:
    python -m slq_mlx.scripts.run_build_db --model Qwen/Qwen3-8B --layer-dir ./quant/Qwen3-8B/4bit
"""

import argparse
import logging
import sys
from pathlib import Path

from slq_mlx.mlx_sensitivity import (
    build_layer_groups,
    parse_group_spec,
    run_build_database,
)
from slq_mlx.mlx_weight_store import ErrorDatabase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Upstream scripts/run_build_database.sh groups only the projections vLLM fuses:
# SwiGLU (gate/up) and QKV (k/q/v). Grouping is per transformer block, so the
# remaining projections (o_proj, mlp.down_proj) stay singletons -- 112 groups on
# a 28-layer model. MLX does not fuse them, so '' (per-layer, 196 groups) is the
# MLX-native choice; we default to upstream's spec so numbers stay comparable.
UPSTREAM_LAYER_GROUPS = "mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj"


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
    parser.add_argument("--layer-groups", default=UPSTREAM_LAYER_GROUPS,
                        help="Groups of layers constrained to share a bitwidth (space-separated groups, "
                             "comma-separated groups of comma-separated patterns). Groups are formed PER "
                             "BLOCK, so anything matching no pattern stays a singleton. Default is "
                             "upstream's spec, which exists because vLLM fuses SwiGLU (gate/up) and QKV "
                             "(k/q/v) and those must share a bitwidth at serving time -- 112 groups on a "
                             "28-layer model. MLX does not fuse them, so pass '' for a genuinely per-layer "
                             "allocation (196 groups) instead.")
    parser.add_argument("--allow-expensive-sensitivity", action="store_true",
                        help="Run linear/Shapley sensitivity even when it is predicted to take many hours. "
                             "Cost scales with the number of groups: 112 groups is ~99 h of Shapley and 196 "
                             "is ~172 h at 10.5 s per metric call. Only meaningful for small models or with "
                             "a reduced --num-permutations.")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()

    if not args.allow_expensive_sensitivity:
        n_groups = len(build_layer_groups(
            ErrorDatabase.load(str(Path(args.layer_dir) / "error_db.json")).layer_names,
            parse_group_spec(args.layer_groups) if args.layer_groups.strip() else [],
        ))
        if n_groups > 50 and not (args.skip_linear and args.skip_shapley):
            n_bw = len([b for b in (args.shapley_bitwidths or "2,3,4,5,6,7,8").split(",") if int(b) != 8])
            n_calls = n_bw * args.num_permutations * (1 + n_groups)
            parser.error(
                f"{n_groups} groups makes sensitivity estimation impractical: Shapley would run "
                f"{n_calls} metric evaluations (~{n_calls * 10.5 / 3600:.0f} h at 10.5 s each). "
                f"The heuristic allocator (run_search_dl.py --method heuristic) needs only "
                f"error_db.json and is the default. Either pass --skip-linear --skip-shapley to "
                f"build just the error database, or --allow-expensive-sensitivity to proceed anyway."
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