"""SLQ bitwidth allocation: ILP + binary search for DL (EAR) and TL targets.

Ported from SLQ slq_search_dl.py and slq_search_tl.py
"""

import argparse
import dataclasses
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import mlx.core as mx
import mlx.nn as nn

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logging.warning("pulp not installed; ILP solver unavailable. Install with: pip install pulp")

from .mlx_data import prepare_calibration_data, get_tokenizer
from .mlx_metrics import compute_baseline_topk, compute_kl_and_ear
from .mlx_sensitivity import LinearSensitivityData, ShapleySensitivityData, load_model
from .mlx_weight_store import WeightStore, ErrorDatabase, LayerGroup, compute_layer_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SOLVER_TIME_LIMIT_SEC = 600
GUARDRAIL_MAX_DRIFT = 2.0
Sensitivity = Union[LinearSensitivityData, ShapleySensitivityData]


def parse_bitwidth_map(s: str) -> Dict[int, float]:
    """Parse '4:4.156,8:8.156' into {nominal_bits: actual_bits}."""
    return {int(k): float(v) for k, v in (p.split(":") for p in s.split(",") if p.strip())}


def get_actual_bitwidth(nominal: int, bitwidth_map: Dict[int, float]) -> float:
    return bitwidth_map.get(nominal, float(nominal))


def load_sensitivity(filepath: str, cls):
    """Load sensitivity database as dataclass."""
    with open(filepath) as f:
        d = json.load(f)

    if d.get("layer_groups") and not d.get("per_layer_normalized", False):
        raise ValueError(f"{filepath}: per_layer_normalized is false; rebuild with mlx_sensitivity.py")

    for k, v in d.items():
        if k.endswith("_per_level") and v:
            d[k] = {n: {int(b): c for b, c in bv.items()} for n, bv in v.items()}

    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def find_database(layer_dir: Path, prefix: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    candidates = sorted(layer_dir.glob(f"{prefix}_*.json"))
    if len(candidates) != 1:
        names = ", ".join(c.name for c in candidates) or "none"
        raise ValueError(f"Need exactly one {prefix}_*.json in {layer_dir} (found: {names}); pick one with --{prefix}-db")
    logger.info(f"Using {prefix} database: {candidates[0]}")
    return str(candidates[0])


def extract_layer_groups(sens: Sensitivity) -> Optional[List[LayerGroup]]:
    if sens.layer_groups:
        return [LayerGroup(**g) for g in sens.layer_groups]
    return None


def layer_costs(error_db: ErrorDatabase, sens: Sensitivity, metric: str) -> Dict[str, Dict[int, float]]:
    """Per-layer cost table c[name][bw] for metric 'kl' or 'ear' (EAR loss)."""
    if isinstance(sens, ShapleySensitivityData):
        return getattr(sens, f"{metric}_shapley_per_level") or {}
    alphas = getattr(sens, f"{metric}_alphas")
    return {n: {b: e * alphas.get(n, 0.0) for b, e in errs.items()} for n, errs in error_db.layers.items()}


def solve_allocation_heuristic(
    error_db: ErrorDatabase,
    target_bitwidth: float,
    available_bw: List[int],
    bitwidth_map: Optional[Dict[int, float]] = None,
    layer_groups: Optional[List[LayerGroup]] = None,
) -> Dict[str, int]:
    """Greedy per-layer allocation (default allocator).

    Start every layer at max bitwidth, then repeatedly take the cheapest
    one-rung reduction until the parameter-weighted average reaches the target.
    A rung is a single step down the available bitwidths, so a layer can travel
    8 -> 7 -> 5 -> 3 across successive steps. Layers in the same group move
    together.
    """
    bitwidth_map = bitwidth_map or {}
    groups = layer_groups or [LayerGroup(i, [n], n) for i, n in enumerate(error_db.layer_names)]
    costs = {n: {b: e for b, e in errs.items()} for n, errs in error_db.layers.items()}
    param_counts = error_db.param_counts

    ordered = sorted(available_bw)
    max_bw = ordered[-1]
    next_lower = {ordered[i]: ordered[i - 1] for i in range(1, len(ordered))}

    def avg_bw(cfg: Dict[str, int]) -> float:
        total = sum(param_counts.get(n, 1) for n in cfg)
        bits = sum(get_actual_bitwidth(cfg[n], bitwidth_map) * param_counts.get(n, 1) for n in cfg)
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
            d_err = d_bits = 0.0
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
                f"Target {target_bitwidth} unreachable: every layer is already at the "
                f"minimum bitwidth, average is {avg_bw(config):.4f}"
            )
            break

        _, group, nxt = best
        group_bw[group.group_id] = nxt
        for n in group.layer_names:
            config[n] = nxt
        steps += 1
        if steps % 100 == 0 or avg_bw(config) <= target_bitwidth:
            logger.info(f"  greedy step {steps}: average bitwidth {avg_bw(config):.4f}")

    logger.info(f"Heuristic allocation converged in {steps} greedy steps")
    return config


def run_heuristic_search(
    args,
    error_db: ErrorDatabase,
    available_bw: List[int],
    bitwidth_map: Dict[int, float],
    groups: Optional[List[LayerGroup]],
    header: Dict[str, object],
) -> Tuple[Dict[str, int], bool]:
    """Default path: per-layer greedy allocation over error_db.json.

    Needs no sensitivity database, so it is the only allocator that is cheap
    enough to use by default. Measured better than the grouped ILP at equal
    budget on MLX (EAR 0.9116 vs 0.9072 at ~4.5 bpw on Qwen3-1.7B) because the
    upstream grouping restricts the ILP to 2 bitwidths.
    """
    config = solve_allocation_heuristic(
        error_db, args.target_bitwidth, available_bw, bitwidth_map, groups
    )

    # Validate against the BF16 baseline: capture it BEFORE apply_config.
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)
    cal_data, masks = load_calibration(args, tokenizer)
    topk_vals, topk_idx = compute_baseline_topk(model, cal_data, args.batch_size, args.top_k)
    apply_config(model, config, WeightStore(args.layer_dir).load())
    kl, ear, n = compute_kl_and_ear(model, topk_vals, topk_idx, cal_data, args.batch_size, masks)
    logger.info(
        f"Heuristic allocation at {args.target_bitwidth:.2f} bpw: "
        f"EAR {ear:.4f}, KL {kl:.4f} (n={n})"
    )

    ok = True if args.min_ear is None else ear >= args.min_ear
    if args.min_ear is not None and not ok:
        logger.warning(f"EAR {ear:.4f} is below --min-ear {args.min_ear}")
    return config, ok


def solve_allocation(
    error_db: ErrorDatabase,
    sens: Sensitivity,
    target_bitwidth: float,
    available_bw: List[int],
    kl_weight: float = 1.0,
    ear_weight: float = 0.0,
    bitwidth_map: Optional[Dict[int, float]] = None,
    layer_groups: Optional[List[LayerGroup]] = None,
) -> Dict[str, int]:
    """Multiple-choice knapsack ILP: one bitwidth per group, average bits <= target."""
    if not HAS_PULP:
        raise RuntimeError("pulp not installed. Run: pip install pulp")

    bitwidth_map = bitwidth_map or {}
    groups = layer_groups or [LayerGroup(i, [n], n) for i, n in enumerate(error_db.layer_names)]

    # Cost tables for weighted metrics
    tables = [(w, layer_costs(error_db, sens, m)) for m, w in (("kl", kl_weight), ("ear", ear_weight)) if w]

    def group_cost(group: LayerGroup, bw: int) -> float:
        return sum(w * t.get(n, {}).get(bw, 0.0) for w, t in tables for n in group.layer_names)

    # Create ILP problem
    prob = pulp.LpProblem("SLQ_Allocation", pulp.LpMinimize)

    # Decision variables: x[group_idx][bw_idx] = 1 if group gets bitwidth.
    # pulp 4.x creates variables only through LpProblem.add_variable(); the old
    # LpVariable(name, cat=...) constructor no longer exists.
    x = {}
    for i, group in enumerate(groups):
        for j, bw in enumerate(available_bw):
            x[(i, j)] = prob.add_variable(f"x_{i}_{j}", cat="Binary")

    # Each group gets exactly one bitwidth
    for i in range(len(groups)):
        prob += pulp.lpSum(x[(i, j)] for j in range(len(available_bw))) == 1

    # Average bitwidth constraint
    params = [sum(error_db.param_counts.get(n, 1) for n in group.layer_names) for group in groups]
    prob += pulp.lpSum(
        x[(i, j)] * get_actual_bitwidth(bw, bitwidth_map) * params[i]
        for i in range(len(groups)) for j, bw in enumerate(available_bw)
    ) <= target_bitwidth * sum(params)

    # Objective: minimize weighted cost
    prob += pulp.lpSum(
        x[(i, j)] * group_cost(group, bw)
        for i, group in enumerate(groups) for j, bw in enumerate(available_bw)
    )

    # Solve
    # pulp 4.x: solve() returns LpSolveStats and PULP_CBC_CMD is gone; HiGHS is
    # the bundled Rust-backed solver (needs `pip install highspy`).
    solver = pulp.HiGHS(msg=False, timeLimit=SOLVER_TIME_LIMIT_SEC)
    stats = prob.solve(solver)

    status = stats.status
    if status not in (pulp.LpSolveStatus.Optimal, pulp.LpSolveStatus.TimeLimit,
                      pulp.LpSolveStatus.SolutionLimit):
        raise RuntimeError(
            f"ILP found no solution at avg bitwidth {target_bitwidth:.3f} "
            f"(status {stats.status_str})"
        )

    if status is not pulp.LpSolveStatus.Optimal:
        logger.warning(f"ILP hit a limit; using best feasible solution (status {stats.status_str})")

    # Extract solution
    config = {}
    for i, group in enumerate(groups):
        for j, bw in enumerate(available_bw):
            if pulp.value(x[(i, j)]) > 0.5:
                for n in group.layer_names:
                    config[n] = bw
    return config


def predict_metrics_from_config(
    config: Dict[str, int],
    error_db: ErrorDatabase,
    sens: Sensitivity,
) -> Tuple[float, float]:
    """Predicted (KL, EAR) as sums of per-layer costs."""
    kl_costs = layer_costs(error_db, sens, "kl")
    ear_costs = layer_costs(error_db, sens, "ear")
    kl = sum(kl_costs.get(n, {}).get(b, 0.0) for n, b in config.items())
    ear = 1.0 - sum(ear_costs.get(n, {}).get(b, 0.0) for n, b in config.items())
    return kl, ear


def compute_avg_bitwidth(config: Dict[str, int], param_counts: Dict[str, int], bitwidth_map: Optional[Dict[int, float]] = None) -> float:
    bitwidth_map = bitwidth_map or {}
    total = sum(param_counts.get(n, 1) for n in config)
    bits = sum(get_actual_bitwidth(b, bitwidth_map) * param_counts.get(n, 1) for n, b in config.items())
    return bits / total if total else 0.0


def distribution_lines(config: Dict[str, int]) -> List[str]:
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


def apply_config(model: nn.Module, config: Dict[str, int], weight_store: WeightStore):
    # Set the weight on the existing layer -- do NOT replace the module.
    # Swapping in a raw array (or a QuantizedLinear) breaks the forward pass
    # ("'mlx.core.array' object is not callable") and changes parameter names.
    from .mlx_model_utils import set_submodule_weight
    for name, bw in config.items():
        weight = weight_store.get_layer_weight(name, bw)
        set_submodule_weight(model, name, weight)


def load_calibration(args, tokenizer) -> Tuple[mx.array, Optional[mx.array]]:
    from .mlx_data import prepare_calibration_data
    return prepare_calibration_data(args.model, args.calibration_data, args.calibration_samples, args.calibration_seq_length, args.seed)


def bisect_budget(
    solve_fn: Callable[[float], Dict[str, int]],
    eval_fn: Callable[[Dict[str, int]], bool],
    satisfies_fn: Callable[[Dict], bool],
    min_bw: float,
    max_bw: float,
    tolerance: float = 0.1,
) -> Tuple[Dict[str, int], Dict]:
    """Binary search for lowest bitwidth satisfying constraint."""
    history = []
    best_config = None
    best_result = None

    lo, hi = min_bw, max_bw
    while hi - lo > tolerance:
        mid = (lo + hi) / 2
        config = solve_fn(mid)
        result = eval_fn(config)
        history.append({"bitwidth": mid, "satisfies": result, "config": config})

        if result:
            best_config = config
            best_result = {"bitwidth": mid, "config": config}
            hi = mid
        else:
            lo = mid

    # If no solution found, return the best feasible at max_bw
    if best_config is None:
        config = solve_fn(max_bw)
        result = eval_fn(config)
        best_config = config
        best_result = {"bitwidth": max_bw, "satisfies": result, "config": config}

    return best_config, best_result


# ===== Distribution-Lossless Search =====

def run_distribution_lossless(
    args,
    error_db: ErrorDatabase,
    sens: Sensitivity,
    solve_fn: Callable[[float], Dict[str, int]],
    bw_range: Tuple[float, float],
    header: Dict[str, object],
) -> Tuple[Dict[str, int], bool]:
    """Search for lowest bitwidth meeting EAR constraint."""

    if args.target_bitwidth is not None:
        # Fixed target bitwidth mode.
        # The BF16 baseline MUST be captured before apply_config -- computing it
        # afterwards compares the quantized model against itself and reports
        # EAR 1.0 / KL 0.0 regardless of the allocation.
        model = load_model(args.model)
        tokenizer = get_tokenizer(args.model)
        cal_data, masks = load_calibration(args, tokenizer)
        topk_vals, topk_idx = compute_baseline_topk(model, cal_data, args.batch_size, args.top_k)

        config = solve_fn(args.target_bitwidth)
        apply_config(model, config, WeightStore(args.layer_dir).load())
        kl, ear, n = compute_kl_and_ear(model, topk_vals, topk_idx, cal_data, args.batch_size, masks)
        # --target-bitwidth fixes the budget, so there is no EAR threshold to
        # satisfy; --min-ear is optional here and only acts as a report gate.
        ok = True if args.min_ear is None else ear >= args.min_ear
        logger.info(f"EAR at {args.target_bitwidth:.2f} bpw: {ear:.4f} (KL {kl:.4f}, n={n})")
        return config, ok

    # Binary search on EAR
    if args.use_predicted_constraints:
        # Use predicted EAR from cost table (no model inference)
        def eval_fn(config):
            _, ear = predict_metrics_from_config(config, error_db, sens)
            return ear >= args.min_ear
        config, result = bisect_budget(solve_fn, eval_fn, lambda r: r, *bw_range, args.bitwidth_tolerance)
        ok = result.get("satisfies", False)
        return config, ok

    # Measured EAR (requires model inference at each step)
    model = load_model(args.model)
    weight_store = WeightStore(args.layer_dir).load()
    tokenizer = get_tokenizer(args.model)
    cal_data, masks = load_calibration(args, tokenizer)
    topk_vals, topk_idx = compute_baseline_topk(model, cal_data, args.batch_size, args.top_k)

    def eval_fn(config):
        apply_config(model, config, weight_store)
        kl, ear, n = compute_kl_and_ear(model, topk_vals, topk_idx, cal_data, args.batch_size, masks)
        return ear >= args.min_ear

    config, result = bisect_budget(solve_fn, eval_fn, lambda r: r, *bw_range, args.bitwidth_tolerance)
    ok = result.get("satisfies", False)
    return config, ok


# ===== Task-Lossless Search (lm-eval in loop) =====

def run_task_lossless(
    args,
    error_db: ErrorDatabase,
    sens: Sensitivity,
    solve_fn: Callable[[float], Dict[str, int]],
    bw_range: Tuple[float, float],
    header: Dict[str, object],
) -> Tuple[Dict[str, int], bool]:
    """Benchmark-in-the-loop TL search using lm-eval."""
    # This requires lm-eval integration - placeholder for now
    logger.warning("Task-Lossless search requires lm-eval integration (not yet implemented in MLX port)")
    # For now, fall back to predicted constraints
    def eval_fn(config):
        kl, ear = predict_metrics_from_config(config, error_db, sens)
        # Use predicted recovery as proxy
        return ear >= args.min_recovery  # proxy

    config, result = bisect_budget(solve_fn, eval_fn, lambda r: r, *bw_range, args.bitwidth_tolerance)
    ok = result.get("satisfies", False)
    return config, ok


def main():
    parser = argparse.ArgumentParser(description="SLQ bitwidth allocation search (MLX port)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer-dir", required=True)
    parser.add_argument("--mode", choices=["dl", "tl"], default="dl")
    parser.add_argument("--method", choices=["heuristic", "ilp"], default="heuristic",
                        help="Allocator: 'heuristic' (default) is the per-layer greedy allocator over "
                             "error_db.json; 'ilp' is the multiple-choice knapsack over a sensitivity DB.")
    parser.add_argument("--sensitivity-method", choices=["shapley", "linear"], default="shapley")
    parser.add_argument("--shapley-db")
    parser.add_argument("--linear-db")
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--ear-weight", type=float)
    parser.add_argument("--min-ear", type=float)
    parser.add_argument("--target-bitwidth", type=float)
    parser.add_argument("--min-recovery", type=float)
    parser.add_argument("--use-predicted-constraints", action="store_true")
    parser.add_argument("--bitwidths")
    parser.add_argument("--bitwidth-map", default="")
    parser.add_argument("--disable-grouping", action="store_true")
    parser.add_argument("--calibration-data", default="fineweb_edu")
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bitwidth-tolerance", type=float, default=0.1)
    parser.add_argument("--output")

    args = parser.parse_args()

    tl = args.mode == "tl"
    if args.kl_weight is None:
        args.kl_weight = 1.0 if tl else 0.0
    if args.ear_weight is None:
        args.ear_weight = 0.0 if tl else 1.0

    if not tl and (args.min_ear is None) == (args.target_bitwidth is None):
        parser.error("--mode dl needs exactly one of --min-ear or --target-bitwidth")

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    layer_dir = Path(args.layer_dir)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)

    error_db = ErrorDatabase.load(str(layer_dir / "error_db.json"))

    available_bw = error_db.get_common_bitwidths()
    if args.bitwidths:
        available_bw = [b for b in available_bw if b in {int(s) for s in args.bitwidths.split(",")}]
    if not available_bw:
        parser.error("no requested bitwidth is present in error_db.json")

    bw_range = tuple(get_actual_bitwidth(f(available_bw), bitwidth_map) for f in (min, max))
    if args.target_bitwidth is not None and args.target_bitwidth <= bw_range[0]:
        parser.error(f"--target-bitwidth must exceed minimum available bitwidth {bw_range[0]}")

    logger.info(f"Available bitwidths: {available_bw}, bitwidth map: {bitwidth_map or 'none'}")

    header: Dict[str, object] = {}

    # ---- Default path: per-layer greedy heuristic over error_db.json ----
    # No sensitivity database needed. Measured better than the grouped ILP at
    # equal budget on MLX, because the upstream grouping restricts the ILP to
    # one bitwidth per group (2 groups = 2 bitwidths total).
    if args.method == "heuristic":
        groups = None if args.disable_grouping else None
        config, ok = run_heuristic_search(
            args, error_db, available_bw, bitwidth_map, groups, header
        )
        avg_bw = compute_avg_bitwidth(config, error_db.param_counts, bitwidth_map)
        tag = f"heuristic_target{args.target_bitwidth:.2f}" if args.target_bitwidth else f"heuristic_ear{args.min_ear}"
        header.update({
            "Model": args.model,
            "Mode": args.mode,
            "Method": "heuristic",
            "Available bitwidths": available_bw,
            "Bitwidth map": bitwidth_map or "none",
            "Layer groups": "per-layer",
            "Average bitwidth": f"{avg_bw:.4f}",
            "Satisfies constraints": ok,
        })
        logger.info("Bitwidth distribution: " + ", ".join(distribution_lines(config)))
        logger.info(f"Average bitwidth {avg_bw:.3f}, satisfies constraints: {ok}")
        out = Path(args.output or layer_dir / f"config_{args.mode}_bws{'-'.join(map(str, available_bw))}_{tag}_bw{avg_bw:.2f}_seed{args.seed}.txt")
        write_config(out, header, config)
        return 0 if ok else 1

    method = args.sensitivity_method
    cls = ShapleySensitivityData if method == "shapley" else LinearSensitivityData
    sens = load_sensitivity(find_database(layer_dir, method, getattr(args, f"{method}_db")), cls)

    if isinstance(sens, ShapleySensitivityData):
        available_bw = [b for b in available_bw if b in sens.bitwidths]
    if not available_bw:
        parser.error("no requested bitwidth is in both --layer-dir and the sensitivity database")

    groups = None if args.disable_grouping else extract_layer_groups(sens)

    def solve(budget):
        if not error_db.param_counts:
            raise ValueError("error_db.json has no param_counts")
        return solve_allocation(error_db, sens, budget, available_bw, args.kl_weight, args.ear_weight, bitwidth_map, groups)

    header.update({
        "Model": args.model,
        "Mode": args.mode,
        "Sensitivity method": method,
        "Available bitwidths": available_bw,
        "Bitwidth map": bitwidth_map or "none",
        "Layer groups": len(groups) if groups else "none",
        "Weights": f"kl={args.kl_weight}, ear={args.ear_weight}",
    })

    if tl:
        config, ok = run_task_lossless(args, error_db, sens, solve_fn=solve, bw_range=bw_range, header=header)
        tag = f"rec{args.min_recovery}_cal{args.calibration_bitwidth}bit"
    elif args.target_bitwidth is not None:
        config, ok = run_distribution_lossless(args, error_db, sens, solve_fn=solve, bw_range=bw_range, header=header)
        tag = f"{method}_target{args.target_bitwidth:.2f}"
    else:
        config, ok = run_distribution_lossless(args, error_db, sens, solve_fn=solve, bw_range=bw_range, header=header)
        tag = f"{method}_{'predicted' if args.use_predicted_constraints else 'measured'}_ear{args.min_ear}"

    bws = "-".join(map(str, available_bw))
    avg_bw = compute_avg_bitwidth(config, error_db.param_counts, bitwidth_map)
    output_path = Path(args.output or layer_dir / f"config_{args.mode}_bws{bws}_{tag}_bw{avg_bw:.2f}{'_grouped' if groups else ''}_seed{args.seed}.txt")

    header.update({"Average bitwidth": f"{avg_bw:.4f}", "Satisfies constraints": ok})
    logger.info("Bitwidth distribution: " + ", ".join(distribution_lines(config)))
    logger.info(f"Average bitwidth {avg_bw:.3f}, satisfies constraints: {ok}")

    write_config(output_path, header, config)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())