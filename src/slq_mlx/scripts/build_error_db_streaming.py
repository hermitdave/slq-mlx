#!/usr/bin/env python3
"""
Streaming / layer-by-layer Hessian error_db builder for SLQ.

Two-phase approach:
  Phase 1: Load a small RTN quant (8-bit) to capture per-layer activations.
  Phase 2: Load BF16 weights one tensor at a time from safetensors and
           measure quant error against the pre-captured Hessian diagonals.

Usage:
    python -m slq_mlx.scripts.build_error_db_streaming \
        --source ~/Projects/ml/models/Ornith-1.5-9B \
        --rtn-source ~/Projects/ml/models/Ornith-1.5-9B-RTN-8bit \
        --output error_db.json \
        --calibration /tmp/ornith_calib_fineweb.npy \
        --bitwidths 2,3,4,6,8 \
        --group-size 64
"""

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------
class _CaptureLinear(nn.Linear):
    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


class _CaptureQuantized(nn.QuantizedLinear):
    def __init__(self, source: nn.QuantizedLinear):
        super().__init__(
            input_dims=source.weight.shape[1],
            output_dims=source.weight.shape[0],
            bias=getattr(source, "bias", None) is not None,
            group_size=getattr(source, "group_size", 64),
            bits=getattr(source, "bits", 4),
            mode=getattr(source, "mode", "affine"),
        )
        for k in source.keys():
            self[k] = source[k]

    def __call__(self, x):
        self._last_input = x
        return super().__call__(x)


def wrap_linear_layers(model):
    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, nn.QuantizedLinear):
            parent_path, _, child_name = name.rpartition(".")
            parent = model
            for part in parent_path.split("."):
                if part:
                    parent = getattr(parent, part)
            cap = _CaptureLinear(module.input_dims, module.output_dims, module.bias is not None)
            cap.weight = module.weight
            setattr(parent, child_name, cap)
            wrapped.append(name)
        elif isinstance(module, nn.QuantizedLinear):
            parent_path, _, child_name = name.rpartition(".")
            parent = model
            for part in parent_path.split("."):
                if part:
                    parent = getattr(parent, part)
            cap = _CaptureQuantized(module)
            setattr(parent, child_name, cap)
            wrapped.append(name)
    return wrapped


def _collect_activations(model, wrapped_names, calibration, batch_size=4):
    hessian_map = {}
    for name in wrapped_names:
        hessian_map[name] = []
    n_samples = calibration.shape[0]
    for i in range(0, n_samples, batch_size):
        batch = mx.array(calibration[i : i + batch_size])
        model(batch)
        for name in wrapped_names:
            module = model
            for part in name.split("."):
                if part:
                    module = getattr(module, part)
            x = module._last_input
            # Explicit copy to break any view that retains the parent buffer.
            x = mx.array(x)
            x_np = np.array(x.astype(mx.float32))
            ss = np.sum(x_np ** 2, axis=tuple(range(x_np.ndim - 1)))
            hessian_map[name].append(ss)
    mx.eval(mx.zeros(1))
    result = {}
    for name, vals in hessian_map.items():
        if vals:
            result[name] = np.mean(np.concatenate(vals, axis=0))
    return result


# ---------------------------------------------------------------------------
# Error measurement
# ---------------------------------------------------------------------------
def _measure_error(bf16_weight, qweight, scales, hessian_diag, group_size=64):
    w = bf16_weight.astype(mx.float32)
    q = qweight.astype(mx.float32)
    s = scales.astype(mx.float32)
    h = hessian_diag.astype(mx.float32)
    wq = (q * s).T
    diff = w - wq
    err = float(mx.sum(diff * diff * h) / (mx.sum(w * w * h) + 1e-12))
    return err


def main():
    ap = argparse.ArgumentParser(description="SLQ streaming error DB builder")
    ap.add_argument("--source", required=True, help="BF16 source model dir")
    ap.add_argument("--rtn-source", required=True, help="8-bit RTN model dir")
    ap.add_argument("--output", required=True, help="Output error_db.json path")
    ap.add_argument("--calibration", required=True, help=".npy calibration data")
    ap.add_argument("--bitwidths", default="2,3,4,6,8", help="Comma-separated bitwidths")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    source = Path(args.source)
    rtn_source = Path(args.rtn_source)
    output = Path(args.output)
    bitwidths = [int(b) for b in args.bitwidths.split(",")]

    # Phase 1: capture activations from RTN model
    print("[phase 1] loading RTN quant for activation capture ...")
    from mlx_lm.utils import load_model as mlx_load_model
    rtn_model, _ = mlx_load_model(rtn_source, lazy=False, strict=False)
    rtn_model.eval()
    wrapped = wrap_linear_layers(rtn_model)
    print(f"  Wrapped {len(wrapped)} linear layers")

    calibration = np.load(args.calibration)
    print(f"  Running calibration (batch_size={args.batch_size}) ...")
    hessian_map = _collect_activations(rtn_model, wrapped, calibration, batch_size=args.batch_size)
    print(f"  Captured Hessian diagonals for {len(hessian_map)} layers")
    del rtn_model
    mx.eval(mx.zeros(1))

    # Phase 2: per-tensor error measurement
    print("[phase 2] measuring per-tensor quant error ...")
    shards = sorted(source.glob("*.safetensors"))
    shard_map = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                shard_map[key] = shard

    db = {"layers": {}, "param_counts": {}}
    all_errors = 0

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = [k for k in f.keys() if k.endswith(".weight") and "mtp" not in k]
            for key in keys:
                base = key[:-7]
                if base not in hessian_map:
                    continue
                w = f.get_tensor(key)
                param_count = int(np.prod(w.shape))
                db["param_counts"][base] = param_count
                layer_errors = {}
                for bw in bitwidths:
                    # Simulate quantize→dequant
                    q, s = mx.quantize(mx.array(w), group_size=args.group_size, bits=bw)
                    q_np = np.array(q.astype(mx.int8))
                    s_np = np.array(s.astype(mx.float32))
                    err = _measure_error(w, q_np, s_np, hessian_map[base], args.group_size)
                    layer_errors[str(bw)] = {"error": err}
                    all_errors += 1
                db["layers"][base] = layer_errors

    output.write_text(json.dumps(db, indent=2))
    print(f"\nDone. Wrote {output}")
    print(f"  Layers with errors: {len(db['layers'])}")
    print(f"  Total measurements: {all_errors}")


if __name__ == "__main__":
    main()
