"""Weight store for SLQ quantized layers.

Ported from SLQ src/utils/weight_store.py
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import mlx.core as mx


@dataclass
class LayerGroup:
    """Layers constrained to share a bitwidth (e.g. fused QKV)."""
    group_id: int
    layer_names: List[str]
    pattern: str


class WeightStore:
    """Stores and loads per-layer quantized weights at multiple bitwidths."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._layer_names: Optional[List[str]] = None
        self._available_bitwidths: Dict[str, List[int]] = {}
        self._qparams_cache: Dict[Tuple[str, int], Tuple[mx.array, mx.array, mx.array]] = {}

    def load(self) -> "WeightStore":
        """Scan directory and index available layers and bitwidths."""
        self._layer_names = []
        self._available_bitwidths = {}

        for entry in os.listdir(self.base_dir):
            layer_dir = os.path.join(self.base_dir, entry)
            if not os.path.isdir(layer_dir):
                continue

            bitwidths = []
            for fname in os.listdir(layer_dir):
                if fname.endswith(".npy"):
                    try:
                        bw = int(fname[:-4])
                        bitwidths.append(bw)
                    except ValueError:
                        pass

            if bitwidths:
                self._layer_names.append(entry)
                self._available_bitwidths[entry] = sorted(bitwidths)

        return self

    def get_layer_names(self) -> List[str]:
        if self._layer_names is None:
            self.load()
        return self._layer_names

    def get_available_bitwidths(self, layer_name: str) -> List[int]:
        if self._available_bitwidths is None:
            self.load()
        return self._available_bitwidths.get(layer_name, [])

    def get_common_bitwidths(self) -> List[int]:
        """Bitwidths available for ALL layers."""
        if not self._available_bitwidths:
            self.load()
        if not self._layer_names:
            return []
        common = set(self._available_bitwidths[self._layer_names[0]])
        for name in self._layer_names[1:]:
            common &= set(self._available_bitwidths[name])
        return sorted(common)

    def get_layer_weight(self, layer_name: str, bitwidth: int) -> mx.array:
        """Get fake-quantized weight for layer at bitwidth (dequantized to float32)."""
        # Check cache
        cache_key = (layer_name, bitwidth)
        if cache_key in self._qparams_cache:
            qweight, scales, zeros, group_size = self._qparams_cache[cache_key]
        else:
            qweight, scales, zeros, group_size = self._load_qparams(layer_name, bitwidth)
            self._qparams_cache[cache_key] = (qweight, scales, zeros, group_size)

        # Dequantize to fake-quantized float32
        d_row, d_col = qweight.shape
        gs = group_size
        scale_expanded = mx.repeat(scales, gs, axis=1)[:, :d_col]
        zero_expanded = mx.repeat(zeros, gs, axis=1)[:, :d_col]
        fake_weight = qweight.astype(mx.float32) * scale_expanded + zero_expanded
        return fake_weight

    def get_qparams(self, layer_name: str, bitwidth: int) -> Tuple[mx.array, mx.array, mx.array, int]:
        """Get raw quantization params (qweight, scales, zeros, group_size)."""
        cache_key = (layer_name, bitwidth)
        if cache_key in self._qparams_cache:
            return self._qparams_cache[cache_key]
        qweight, scales, zeros, group_size = self._load_qparams(layer_name, bitwidth)
        self._qparams_cache[cache_key] = (qweight, scales, zeros, group_size)
        return qweight, scales, zeros, group_size

    def _load_qparams(self, layer_name: str, bitwidth: int) -> Tuple[mx.array, mx.array, mx.array, int]:
        """Load qweight, scales, zeros, and group_size from disk."""
        layer_dir = os.path.join(self.base_dir, layer_name)

        # Load qweight
        qweight_path = os.path.join(layer_dir, f"{bitwidth}.npy")
        qweight = mx.array(np.load(qweight_path))

        # Load qparams
        qparams_path = os.path.join(layer_dir, f"{bitwidth}_qparams.json")
        with open(qparams_path, "r") as f:
            qparams = json.load(f)

        scales = mx.array(qparams["scales"])
        zeros = mx.array(qparams["zeros"])
        group_size = qparams.get("group_size", 128)

        return qweight, scales, zeros, group_size


def save_qparams(layer_dir: str, bitwidth: int, qweight: mx.array, scales: mx.array, zeros: mx.array, handle):
    """Save quantization parameters alongside quantized weight."""
    # Save qweight as numpy
    import numpy as np
    np.save(os.path.join(layer_dir, f"{bitwidth}.npy"), np.array(qweight))

    # Save qparams as JSON
    qparams = {
        "scales": np.array(scales).tolist(),
        "zeros": np.array(zeros).tolist(),
        "group_size": handle.group_size,
        "sym": handle.sym,
        "perchannel": handle.perchannel,
        "bitwidth": bitwidth,
        "shape": list(qweight.shape),
    }
    with open(os.path.join(layer_dir, f"{bitwidth}_qparams.json"), "w") as f:
        json.dump(qparams, f, indent=2)


class ErrorDatabase:
    """Per-layer relative squared Frobenius error for every available bitwidth."""

    def __init__(self):
        self.layers: Dict[str, Dict[int, float]] = {}
        self.layer_names: List[str] = []
        self.param_counts: Dict[str, int] = {}

    def add_layer_option(self, layer_name: str, bitwidth: int, error: float):
        if layer_name not in self.layers:
            self.layers[layer_name] = {}
            self.layer_names.append(layer_name)
        self.layers[layer_name][bitwidth] = error

    def get_common_bitwidths(self) -> List[int]:
        if not self.layer_names:
            return []
        common = set(self.layers[self.layer_names[0]].keys())
        for name in self.layer_names[1:]:
            common &= set(self.layers[name].keys())
        return sorted(common)

    def save(self, filepath: str):
        layers_serializable = {
            n: {str(bw): {"error": e} for bw, e in errs.items()}
            for n, errs in self.layers.items()
        }
        data = {
            "layers": layers_serializable,
            "param_counts": self.param_counts,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ErrorDatabase":
        with open(filepath, "r") as f:
            data = json.load(f)
        db = cls()
        for name, opts in data["layers"].items():
            for bw_str, opt in opts.items():
                db.add_layer_option(name, int(bw_str), opt["error"])
        db.param_counts = data.get("param_counts", {})
        return db


def compute_layer_error(original: mx.array, quantized: mx.array) -> float:
    """Relative squared Frobenius error."""
    diff = original.astype(mx.float32) - quantized.astype(mx.float32)
    return float(mx.sum(diff * diff) / mx.sum(original.astype(mx.float32) * original.astype(mx.float32)))