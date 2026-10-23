"""Sensitivity estimation for SLQ: Linear alphas + Shapley values.

Ported from SLQ src/sensitivity/database.py and slq_build_database.py
"""

import hashlib
import json
import logging
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .mlx_metrics import compute_all_metrics, compute_baseline_topk
from .mlx_model_utils import get_layer_weights, get_submodule, set_layer_weights
from .mlx_weight_store import ErrorDatabase, LayerGroup, WeightStore, compute_layer_error
from .mlx_data import prepare_calibration_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LinearSensitivityData:
    """Per-layer linear alphas: cost = alpha * relative weight error."""
    nll_alphas: Dict[str, float] = field(default_factory=dict)
    kl_alphas: Dict[str, float] = field(default_factory=dict)
    ear_alphas: Dict[str, float] = field(default_factory=dict)
    layer_names: List[str] = field(default_factory=list)
    top_k: int = 10
    calibration_dataset: str = ""
    calibration_samples: int = 0
    num_noise_samples: int = 15
    seed: int = 0
    per_layer_normalized: bool = True
    layer_groups: Optional[List[Dict]] = None


@dataclass
class ShapleySensitivityData:
    """Multi-bitwidth Shapley values per player and per layer."""
    nll_shapley_values: Optional[np.ndarray] = None
    kl_shapley_values: Optional[np.ndarray] = None
    ear_shapley_values: Optional[np.ndarray] = None
    nll_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None
    kl_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None
    ear_shapley_per_level: Optional[Dict[str, Dict[int, float]]] = None
    layer_names: List[str] = field(default_factory=list)
    bitwidths: List[int] = field(default_factory=list)
    top_k: int = 10
    calibration_dataset: str = ""
    calibration_samples: int = 0
    num_permutations: int = 50
    seed: int = 0
    per_layer_normalized: bool = True
    layer_groups: Optional[List[Dict]] = None


class GPUWeightCache:
    """Weight cache for Shapley permutation swapping.

    The baseline (every layer at ``max_bw``) is pinned and loaded once; weights at
    the other bitwidths are dequantized on demand and evicted LRU once the cache
    exceeds ``max_cache_bytes``.

    The original implementation preloaded every layer at every bitwidth. That
    OOM-killed on the 3.4 GB Qwen3-1.7B model and would need roughly 150 GB of
    unified memory at 27B -- the Shapley pass only ever holds two bitwidths at a
    time (the baseline plus the target level), so preloading all of them is pure
    waste.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_store: WeightStore,
        bitwidths: List[int],
        preload: bool = True,
        max_cache_bytes: int = 6 * 2**30,
    ):
        self.model = model
        self.weight_store = weight_store
        self.bitwidths = list(bitwidths)
        self.max_bw = max(bitwidths)
        self.max_cache_bytes = max_cache_bytes

        self._pinned: Dict[str, mx.array] = {}
        self._cached: "OrderedDict[Tuple[str, int], mx.array]" = OrderedDict()
        self._cache_bytes = 0
        self.current_bw: Dict[str, int] = {}
        # MLX arrays are immutable value types and weights are replaced by
        # reference, so holding the array is enough -- mx.array has no .copy().
        self.original_weights: Dict[str, mx.array] = {}

        for name in weight_store.get_layer_names():
            self.original_weights[name] = get_submodule(model, name).weight

        if preload:
            self._preload_baseline()

    def _preload_baseline(self):
        """Load every layer at the maximum bitwidth; this is the Shapley baseline."""
        names = list(self.original_weights)
        logger.info(f"Preloading baseline: {len(names)} layers at {self.max_bw}-bit ...")
        for name in names:
            self._pinned[name] = self.weight_store.get_layer_weight(name, self.max_bw)
            self.current_bw[name] = self.max_bw
        mx.eval(list(self._pinned.values()))
        logger.info(f"Baseline resident: {self._bytes(list(self._pinned.values())) / 2**30:.2f} GiB")

    @staticmethod
    def _bytes(arrays) -> int:
        return sum(int(a.nbytes) for a in arrays)

    def _evict_until(self, incoming: int):
        while self._cached and self._cache_bytes + incoming > self.max_cache_bytes:
            _, evicted = self._cached.popitem(last=False)
            self._cache_bytes -= int(evicted.nbytes)
            mx.eval(mx.zeros(1))

    def _get(self, name: str, bitwidth: int) -> mx.array:
        if bitwidth == self.max_bw and name in self._pinned:
            return self._pinned[name]

        key = (name, bitwidth)
        if key in self._cached:
            self._cached.move_to_end(key)
            return self._cached[key]

        weight = self.weight_store.get_layer_weight(name, bitwidth)
        self._evict_until(int(weight.nbytes))
        self._cached[key] = weight
        self._cache_bytes += int(weight.nbytes)
        return weight

    def cache_stats(self) -> str:
        return (f"{len(self._pinned)} pinned, {len(self._cached)} cached, "
                f"{self._cache_bytes / 2**30:.2f} GiB target cache")

    def swap_all_layers(self, model: nn.Module, target_bw: Dict[str, int]):
        """Swap all layers to target bitwidths."""
        for name, bw in target_bw.items():
            self.swap_layer(model, name, bw)

    def swap_layer(self, model: nn.Module, name: str, bitwidth: int):
        """Swap a single layer to target bitwidth."""
        weight = self._get(name, bitwidth)
        set_layer_weights(model, name, weight)
        self.current_bw[name] = bitwidth
        mx.eval(weight)  # Materialize


def build_error_database(model: nn.Module, weight_store: WeightStore) -> ErrorDatabase:
    """Compute relative weight error of every quantized layer vs original."""
    error_db = ErrorDatabase()
    for name in weight_store.get_layer_names():
        original = get_layer_weights(model, name)
        if original is None:
            continue
        error_db.param_counts[name] = original.size
        for bw in weight_store.get_available_bitwidths(name):
            quantized = weight_store.get_layer_weight(name, bw)
            error = compute_layer_error(original, quantized)
            error_db.add_layer_option(name, bw, error)
    return error_db


def estimate_linear_alphas(
    model: nn.Module,
    original_weights: Dict[str, mx.array],
    groups: List[LayerGroup],
    metric_fn,
    num_samples: int = 15,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Per-layer linear alphas: mean of metric degradation / t^2 over Gaussian noise levels t."""
    t_levels = np.linspace(0.01, 0.3, num_samples)

    nll_base, kl_base, ear_base = metric_fn()

    nll_deltas = {name: [] for name in original_weights}
    kl_deltas = {name: [] for name in original_weights}
    ear_deltas = {name: [] for name in original_weights}

    for group in groups:
        for t in t_levels:
            # Add noise to all layers in group
            for name in group.layer_names:
                orig = original_weights[name]
                noise = mx.random.normal(orig.shape) * orig.std() * t
                set_layer_weights(model, name, orig + noise)

            nll, kl, ear = metric_fn()

            for name in group.layer_names:
                nll_deltas[name].append(nll - nll_base)
                kl_deltas[name].append(kl - kl_base)
                ear_deltas[name].append(ear_base - ear)  # EAR loss

            # Restore
            for name in group.layer_names:
                set_layer_weights(model, name, original_weights[name])

    # Compute alphas: mean(delta / t^2) per layer
    nll_alphas, kl_alphas, ear_alphas = {}, {}, {}
    for group in groups:
        for name in group.layer_names:
            nll_alpha = np.mean(np.array(nll_deltas[name]) / t_levels**2).clip(min=0)
            kl_alpha = np.mean(np.array(kl_deltas[name]) / t_levels**2).clip(min=0)
            ear_alpha = np.mean(np.array(ear_deltas[name]) / t_levels**2).clip(min=0)

            # Normalize by group size (Eqs 5-6)
            nll_alpha /= len(group.layer_names)
            kl_alpha /= len(group.layer_names)
            ear_alpha /= len(group.layer_names)

            nll_alphas[name] = float(nll_alpha)
            kl_alphas[name] = float(kl_alpha)
            ear_alphas[name] = float(ear_alpha)

    return nll_alphas, kl_alphas, ear_alphas


def estimate_shapley_values(
    model: nn.Module,
    gpu_cache: GPUWeightCache,
    groups: List[LayerGroup],
    bitwidths: List[int],
    metric_fn,
    num_permutations: int = 50,
) -> np.ndarray:
    """Algorithm 1: Shapley values for [NLL, KL, EAR] x groups x bitwidths."""
    max_bw = max(bitwidths)
    shapley = np.zeros((3, len(groups), len(bitwidths)))
    sign = np.array([1.0, 1.0, -1.0])  # NLL, KL increase; EAR decreases

    for level_idx, target_bw in enumerate(bitwidths):
        if target_bw == max_bw:
            continue

        for _ in range(num_permutations):
            perm = np.random.permutation(len(groups))

            # Start with all layers at max_bw
            gpu_cache.swap_all_layers(model, {n: max_bw for g in groups for n in g.layer_names})
            prev = np.array(metric_fn())

            for g_idx in perm:
                for name in groups[g_idx].layer_names:
                    gpu_cache.swap_layer(model, name, target_bw)

                cur = np.array(metric_fn())
                shapley[:, g_idx, level_idx] += sign * (cur - prev) / num_permutations
                prev = cur

            mx.eval(mx.zeros(1))  # Clear cache

    return shapley


def per_layer_values(values: np.ndarray, groups: List[LayerGroup], bitwidths: List[int]) -> Dict[str, Dict[int, float]]:
    """Each layer stores phi_group / |group| so summing over group counts once."""
    return {
        name: {bw: float(values[i, j]) / len(g.layer_names) for j, bw in enumerate(bitwidths)}
        for i, g in enumerate(groups)
        for name in g.layer_names
    }


def save_json(data, filepath: Path):
    """Save dataclass as JSON."""
    d = asdict(data)
    # Convert numpy arrays to lists
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
    with open(filepath, "w") as f:
        json.dump(d, f, indent=2)
    logger.info(f"Saved {filepath}")


def parse_group_spec(spec: str) -> List[List[str]]:
    """Parse layer group specification string.

    Format: "pattern1,pattern2 pattern3,pattern4" (space-separated groups, comma-separated patterns)
    """
    if not spec.strip():
        return []
    groups = []
    for group_str in spec.split():
        patterns = [p.strip() for p in group_str.split(",")]
        groups.append(patterns)
    return groups


_BLOCK_RE = re.compile(r"layers\.(\d+)\.")


def build_layer_groups(layer_names: List[str], group_patterns: List[List[str]]) -> List[LayerGroup]:
    """Per-block grouping, matching upstream ``src/models/grouping.py``.

    Layers matching the same pattern group *within one transformer block* form a
    single group; anything matching no pattern stays a singleton. Grouping
    globally (one group per pattern-set across all blocks) would collapse 28
    blocks into one group and leave the sensitivity DB with only a couple of
    distinct cost profiles -- the ILP then has almost nothing to work with.

    Upstream's docstring is explicit that this exists for "vLLM-fused
    projections": vLLM fuses SwiGLU (gate/up) and QKV (k/q/v), so those must
    share a bitwidth at serving time. MLX does not fuse them, so a caller
    targeting MLX should pass a per-layer spec (or none) instead.
    """
    grouped: Dict[Tuple[int, int], List[str]] = {}
    singles: List[str] = []
    for name in layer_names:
        m = _BLOCK_RE.search(name)
        g = next(
            (gi for gi, pats in enumerate(group_patterns)
             if m and any(p in name[m.end():] for p in pats)),
            None,
        )
        if g is None:
            singles.append(name)
        else:
            grouped.setdefault((int(m.group(1)), g), []).append(name)

    units = [
        (sorted(names), f"block_{b}:{','.join(group_patterns[g])}")
        for (b, g), names in sorted(grouped.items())
    ]
    units += [([name], name) for name in singles]
    return [LayerGroup(i, names, pattern) for i, (names, pattern) in enumerate(units)]


def run_build_database(
    model_id: str,
    layer_dir: str,
    calibration_data: str = "fineweb_edu",
    calibration_samples: int = 512,
    calibration_seq_len: int = 2048,
    batch_size: int = 4,
    top_k: int = 10,
    seed: int = 42,
    num_noise_samples: int = 15,
    skip_linear: bool = False,
    shapley_bitwidths: str = "2,3,4,5,6,7,8",
    num_permutations: int = 50,
    layer_groups: str = "",
    force_recompute: bool = False,
) -> Tuple[str, str]:
    """Run sensitivity database building.

    Returns paths to (linear_db_path, shapley_db_path).

    ``layer_groups=""`` means per-layer: every layer becomes its own group, so
    the allocation is genuinely per-layer. This is the MLX-appropriate default
    because MLX does not fuse SwiGLU or QKV -- the upstream grouping exists only
    so vLLM can serve fused projections at a shared bitwidth.

    Per-layer sensitivity estimation is expensive, though: cost scales as
    ``len(groups) * num_levels`` metric calls for linear and
    ``len(bitwidths) * num_perms * (1 + len(groups))`` for Shapley. At 196
    layers that is ~9 h (linear) and ~172 h (Shapley) at 10.5 s per call, so
    ``run_build_db.py`` refuses per-layer sensitivity unless explicitly allowed.
    """
    mx.random.seed(seed)
    np.random.seed(seed)

    layer_dir = Path(layer_dir)
    use_grouping = bool(layer_groups.strip())
    group_suffix = f"_grp{hashlib.md5(layer_groups.encode()).hexdigest()[:8]}" if use_grouping else ""
    tag = f"topk{top_k}_{calibration_data}_{calibration_samples}x{calibration_seq_len}"
    bitwidths = sorted(int(b) for b in shapley_bitwidths.split(","))

    error_db_path = layer_dir / "error_db.json"
    baseline_path = layer_dir / f"baseline_{tag}_seed{seed}.npz"
    linear_path = layer_dir / f"linear_{tag}_{num_noise_samples}noise{group_suffix}_seed{seed}.json"
    shapley_path = layer_dir / f"shapley_{tag}_{num_permutations}perms_bw{'-'.join(map(str, bitwidths))}{group_suffix}_seed{seed}.json"

    weight_store = WeightStore(str(layer_dir)).load()
    model = load_model(model_id)  # Need to implement
    tokenizer = get_tokenizer(model_id)

    # Prepare calibration data
    input_ids, masks = prepare_calibration_data(model_id, calibration_data, calibration_samples, calibration_seq_len, seed)

    # Error database
    if error_db_path.exists() and not force_recompute:
        error_db = ErrorDatabase.load(str(error_db_path))
    else:
        error_db = build_error_database(model, weight_store)
        error_db.save(str(error_db_path))

    # Baseline top-K
    if baseline_path.exists() and not force_recompute:
        baseline = np.load(baseline_path)
        topk_vals, topk_idx = baseline["topk_vals"], baseline["topk_idx"]
    else:
        topk_vals, topk_idx = compute_baseline_topk(model, input_ids, batch_size, top_k)
        np.savez(str(baseline_path), topk_vals=np.array(topk_vals), topk_idx=np.array(topk_idx))

    def metric_fn():
        return compute_all_metrics(model, input_ids, topk_vals, topk_idx, batch_size, masks)

    group_patterns = parse_group_spec(layer_groups) if use_grouping else []
    # With no patterns every layer falls through to a singleton group, so ''
    # means per-layer -- matching upstream's "'' for per-layer".
    groups = build_layer_groups(error_db.layer_names, group_patterns)
    logger.info(f"{len(error_db.layer_names)} layers in {len(groups)} groups")

    meta = dict(
        layer_names=error_db.layer_names,
        top_k=top_k,
        calibration_dataset=calibration_data,
        calibration_samples=calibration_samples,
        seed=seed,
        per_layer_normalized=True,
        layer_groups=[asdict(g) for g in groups] if use_grouping else None,
    )

    # Linear alphas
    if not skip_linear and (force_recompute or not linear_path.exists()):
        # MLX arrays are immutable value types; weights are swapped by reference,
        # so holding the array is sufficient (mx.array has no .copy()).
        original_weights = {n: get_layer_weights(model, n) for n in error_db.layer_names}
        nll_a, kl_a, ear_a = estimate_linear_alphas(model, original_weights, groups, metric_fn, num_noise_samples)
        save_json(LinearSensitivityData(
            nll_alphas=nll_a, kl_alphas=kl_a, ear_alphas=ear_a,
            num_noise_samples=num_noise_samples, **meta
        ), linear_path)

    # Shapley values
    if force_recompute or not shapley_path.exists():
        gpu_cache = GPUWeightCache(model, weight_store, bitwidths, preload=True)
        nll_s, kl_s, ear_s = estimate_shapley_values(model, gpu_cache, groups, bitwidths, metric_fn, num_permutations)
        save_json(ShapleySensitivityData(
            nll_shapley_values=nll_s, kl_shapley_values=kl_s, ear_shapley_values=ear_s,
            nll_shapley_per_level=per_layer_values(nll_s, groups, bitwidths),
            kl_shapley_per_level=per_layer_values(kl_s, groups, bitwidths),
            ear_shapley_per_level=per_layer_values(ear_s, groups, bitwidths),
            bitwidths=bitwidths, num_permutations=num_permutations, **meta
        ), shapley_path)

    return str(linear_path), str(shapley_path)


# Helper functions needed - import from appropriate modules
def load_model(model_id: str) -> nn.Module:
    """Load model for calibration."""
    from mlx_lm import load
    model, _ = load(model_id)
    return model


def get_tokenizer(model_id: str):
    from .mlx_data import get_tokenizer
    return get_tokenizer(model_id)