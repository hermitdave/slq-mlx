#!/usr/bin/env python3
"""Simple SLQ search using only error database with heuristic allocation.

Usage:
    python scripts/run_simple_search.py --model Qwen/Qwen3-1.7B --layer-dir ./quant/Qwen3-1.7B/4bit --target-bits 4.5
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

from slq_mlx.mlx_weight_store import WeightStore, ErrorDatabase, LayerGroup
from slq_mlx.mlx_data import prepare_calibration_data, get_tokenizer, get_simple_calibration
from slq_mlx.mlx_metrics import compute_baseline_topk, compute_kl_and_ear
from slq_mlx.mlx_model_utils import get_submodule, set_submodule
from slq_mlx.mlx_sensitivity import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_bitwidth_map(s: str) -> Dict[int, float]:
    return {int(k): float(v) for k, v in (p.split(":") for p in s.split(",") if p.strip())}


def get_actual_bitwidth(nominal: int, bitwidth_map: Dict[int, float]) -> float:
    return bitwidth_map.get(nominal, float(nominal))


def find_database(layer_dir: Path, prefix: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    candidates = sorted(layer_dir.glob(f"{prefix}_*.json"))
    if len(candidates) != 1:
        names = ", ".join(c.name for c in candidates) or "none"
        raise ValueError(f"Need exactly one {prefix}_*.json in {layer_dir} (found: {names})")
    logger.info(f"Using {prefix} database: {candidates[0]}")
    return str(candidates[0])


def extract_layer_groups(sens) -> Optional[List[LayerGroup]]:
    if sens.layer_groups:
        return [LayerGroup(**g) for g in sens.layer_groups]
    return None


def layer_costs(error_db: ErrorDatabase, metric: str = "error") -> Dict[str, Dict[int, float]]:
    """Per-layer cost table using relative Frobenius error."""
    return {n: {b: e for b, e in errs.items()} for n, errs in error_db.layers.items()}


def solve_allocation_heuristic(
    error_db: ErrorDatabase,
    target_bitwidth: float,
    available_bw: List[int],
    bitwidth_map: Optional[Dict[int, float]] = None,
    layer_groups: Optional[List[LayerGroup]] = None,
) -> Dict[str, int]:
    """Greedy bitwidth allocation.

    Start every layer at the maximum bitwidth, then repeatedly take the cheapest
    one-rung reduction until the parameter-weighted average reaches the target.

    A rung is a single step down the available bitwidths, so a layer can travel
    8 -> 6 -> 5 -> 3 across successive steps. The previous version only considered
    one jump from max_bw and refused to touch a layer twice, so with more than two
    bitwidths it always picked the shallowest jump (8 -> 6) and returned 100%
    6-bit for a 4.5 bpw target. Layers in the same group move together.
    """
    bitwidth_map = bitwidth_map or {}
    groups = layer_groups or [LayerGroup(i, [n], n) for i, n in enumerate(error_db.layer_names)]
    costs = layer_costs(error_db)
    param_counts = error_db.param_counts

    ordered = sorted(available_bw)
    max_bw = ordered[-1]
    next_lower = {ordered[i]: ordered[i - 1] for i in range(1, len(ordered))}

    def avg_bw(cfg: Dict[str, int]) -> float:
        total = sum(param_counts.get(n, 1) for n in cfg)
        bits = sum(
            get_actual_bitwidth(cfg[n], bitwidth_map) * param_counts.get(n, 1)
            for n in cfg
        )
        return bits / total if total else 0.0

    config = {n: max_bw for n in error_db.layer_names}
    group_bw = {g.group_id: max_bw for g in groups}

    steps = 0
    while avg_bw(config) > target_bitwidth:
        best = None  # (marginal cost per bit saved, group, next bitwidth)
        for group in groups:
            cur = group_bw[group.group_id]
            nxt = next_lower.get(cur)
            if nxt is None:
                continue
            d_err = 0.0
            d_bits = 0.0
            for n in group.layer_names:
                e_cur = costs.get(n, {}).get(cur)
                e_next = costs.get(n, {}).get(nxt)
                if e_cur is None or e_next is None:
                    continue
                p = param_counts.get(n, 1)
                d_err += e_next - e_cur
                d_bits += (get_actual_bitwidth(cur, bitwidth_map)
                           - get_actual_bitwidth(nxt, bitwidth_map)) * p
            if d_bits <= 0:
                continue
            marginal = d_err / d_bits
            if best is None or marginal < best[0]:
                best = (marginal, group, nxt)

        if best is None:
            logger.warning(
                f"Target {target_bitwidth} unreachable: every layer is already at "
                f"the minimum bitwidth, average is {avg_bw(config):.4f}"
            )
            break

        _, group, nxt = best
        group_bw[group.group_id] = nxt
        for n in group.layer_names:
            config[n] = nxt
        steps += 1
        if steps % 100 == 0 or avg_bw(config) <= target_bitwidth:
            logger.info(f"  step {steps}: average bitwidth {avg_bw(config):.4f}")

    logger.info(f"Allocation converged in {steps} greedy steps")
    return config


def compute_avg_bitwidth(config: Dict[str, int], param_counts: Dict[str, int], bitwidth_map: Optional[Dict[int, float]] = None) -> float:
    bitwidth_map = bitwidth_map or {}
    total = sum(param_counts.get(n, 1) for n in config)
    bits = sum(get_actual_bitwidth(b, bitwidth_map) * param_counts.get(n, 1) for n, b in config.items())
    return bits / total if total else 0.0


def distribution_lines(config: Dict[str, int]) -> List[str]:
    from collections import Counter
    dist = Counter(config.values())
    return [f"{b}-bit: {dist[b]} layers ({100.0 * dist[b] / len(config):.1f}%)" for b in sorted(dist, reverse=True)]


def write_config(path: Path, header: Dict[str, object], config: Dict[str, int]):
    with open(path, "w") as f:
        for k, v in header.items():
            f.write(f"# {k}: {v}\n")
        f.write("#\n")
        f.write("# Bitwidth distribution:\n")
        for line in distribution_lines(config):
            f.write(f"# {line}\n")
        f.write("#\n")
        for n, b in config.items():
            f.write(f"{n}: {b}\n")
    logger.info(f"Config saved to: {path}")


def apply_config(model: nn.Module, config: Dict[str, int], weight_store: WeightStore, available_bw: List[int]):
    """Replace layers with QuantizedLinear modules at specified bitwidths."""
    from slq_mlx.mlx_quantizer import QuantizedLinear
    for name, bw in config.items():
        qweight, scales, zeros, group_size = weight_store.get_qparams(name, bw)
        original_layer = get_submodule(model, name)
        bias = getattr(original_layer, 'bias', None)
        qlayer = QuantizedLinear.from_qparams(qweight, scales, zeros, bias, group_size)
        set_submodule(model, name, qlayer)


def load_calibration(model_id: str, n_samples: int, seq_len: int, seed: int,
                     dataset: str = "fineweb_edu") -> mx.array:
    """Load calibration data for validation.

    Must use the same corpus as quantization -- validating on the synthetic
    fallback while calibrating on FineWeb-Edu reports an EAR that has nothing
    to do with the shipped model.
    """
    input_ids, _ = prepare_calibration_data(model_id, dataset, n_samples, seq_len, seed)
    return input_ids


def main():
    parser = argparse.ArgumentParser(description="Simple SLQ search using error database heuristic")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--target-bits", type=float, required=True, help="Target average bits per parameter")
    parser.add_argument("--bitwidths", default="2,3,4,5,6,8")
    parser.add_argument("--bitwidth-map", default="")
    parser.add_argument("--calibration-data", default="fineweb_edu",
                        choices=["fineweb_edu", "wikitext2", "c4"],
                        help="Must match the corpus used for quantization")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--calibration-seq-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()

    import numpy as np
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    layer_dir = Path(args.layer_dir)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)

    error_db = ErrorDatabase.load(str(layer_dir / "error_db.json"))
    method = "error"
    available_bw = error_db.get_common_bitwidths()
    if args.bitwidths:
        available_bw = [b for b in available_bw if b in {int(s) for s in args.bitwidths.split(",")}]
    if not available_bw:
        parser.error("no requested bitwidth is in both --layer-dir and the error database")

    bw_range = tuple(get_actual_bitwidth(f(available_bw), bitwidth_map) for f in (min, max))
    if args.target_bits <= bw_range[0]:
        parser.error(f"--target-bits must exceed minimum available bitwidth {bw_range[0]}")

    logger.info(f"Available bitwidths: {available_bw}, bitwidth map: {bitwidth_map or 'none'}")
    logger.info(f"Target avg bitwidth: {args.target_bits:.3f}")

    # Load model for validation
    model = load_model(args.model)
    weight_store = WeightStore(args.layer_dir).load()

    # Build config using heuristic
    config = solve_allocation_heuristic(
        error_db, args.target_bits, available_bw, bitwidth_map, None
    )

    # Validate with calibration data
    tokenizer = get_tokenizer(args.model)
    cal_data = load_calibration(args.model, args.calibration_samples,
                                args.calibration_seq_length, args.seed,
                                args.calibration_data)
    topk_vals, topk_idx = compute_baseline_topk(model, cal_data, 4, 10)
    
    apply_config(model, config, weight_store, available_bw)
    kl, ear, n = compute_kl_and_ear(model, topk_vals, topk_idx, cal_data, 4, None)

    avg_bw = compute_avg_bitwidth(config, error_db.param_counts, bitwidth_map)
    logger.info("Bitwidth distribution: " + ", ".join(distribution_lines(config)))
    logger.info(f"Average bitwidth: {avg_bw:.4f}")
    logger.info(f"Validation KL: {kl:.6f}, EAR: {ear:.6f}")

    # Save config
    header = {
        "Model": args.model,
        "Mode": "heuristic",
        "Target bitwidth": args.target_bits,
        "Available bitwidths": available_bw,
        "Bitwidth map": bitwidth_map or "none",
        "Average bitwidth": f"{avg_bw:.4f}",
        "Validation KL": f"{kl:.6f}",
        "Validation EAR": f"{ear:.6f}",
    }
    bws = "-".join(map(str, available_bw))
    output_path = Path(args.output or layer_dir / f"config_heuristic_target{args.target_bits:.2f}_bws{bws}_bw{avg_bw:.2f}_seed{args.seed}.txt")
    write_config(output_path, header, config)
    logger.info(f"Config saved to: {output_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())