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
import torch
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


from slq_mlx.mlx_model_utils import get_submodule, set_submodule


def wrap_linear_layers(model):
    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not isinstance(module, nn.QuantizedLinear):
            parent_path, _, child_name = name.rpartition(".")
            parent = get_submodule(model, parent_path) if parent_path else model
            cap = _CaptureLinear(module.weight.shape[1], module.weight.shape[0], hasattr(module, "bias"))
            cap.weight = module.weight
            set_submodule(model, name, cap)
            wrapped.append(name)
        elif isinstance(module, nn.QuantizedLinear):
            parent_path, _, child_name = name.rpartition(".")
            parent = get_submodule(model, parent_path) if parent_path else model
            cap = _CaptureQuantized(module)
            set_submodule(model, name, cap)
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
            module = get_submodule(model, name)
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
def _measure_error(bf16_weight, qweight, scales, zeros, hessian_diag, group_size, bits):
    # bf16_weight may come from safetensors as a torch tensor or numpy array
    if hasattr(bf16_weight, "detach"):
        bf16_weight = np.asarray(bf16_weight.detach().cpu().to(torch.float32))
    else:
        bf16_weight = np.asarray(bf16_weight)
    w = bf16_weight.astype(np.float32)

    q = np.asarray(qweight)
    s = np.asarray(scales)
    z = np.asarray(zeros) if zeros is not None else np.zeros_like(s)
    h = np.asarray(hessian_diag).astype(np.float32) if hessian_diag is not None else np.ones(w.shape[1])

    # mx.quantize returns packed uint32: [output_dim, packed_cols]
    # Unpack to [output_dim, input_dim] for comparison
    if bits < 32:
        vals_per_word = q.dtype.itemsize * 8 // bits
        total_elements = q.shape[1] * vals_per_word
        q_unpacked = np.zeros((q.shape[0], total_elements), dtype=np.int32)
        mask = (1 << bits) - 1
        for i in range(vals_per_word):
            q_unpacked[:, i::vals_per_word] = ((q >> (bits * i)) & mask).astype(np.int32)
        q = q_unpacked

    # Expand s/z from [output_dim, num_groups] to [output_dim, input_dim]
    repeat = w.shape[1] // s.shape[1]
    if repeat > 1:
        s = np.repeat(s, repeat, axis=1)
        z = np.repeat(z, repeat, axis=1)

    wq = (q.astype(np.float32) * s + z)
    diff = w - wq
    err = float(np.sum(diff * diff * h) / (np.sum(w * w * h) + 1e-12))
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

    # Phase 2: per-tensor error measurement from Stage 1 outputs
    print("[phase 2] measuring per-tensor quant error from Stage 1 outputs ...")
    quant_dir = source.parent / "quant"
    shards = sorted(source.glob("*.safetensors"))
    shard_map = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                shard_map[key] = shard

    db = {"layers": {}, "param_counts": {}}
    all_errors = 0

    for bw in bitwidths:
        bw_dir = quant_dir / f"{bw}bit"
        if not bw_dir.exists():
            print(f"  WARNING: {bw_dir} missing, skipping bitwidth {bw}")
            continue
        for layer_dir in sorted(bw_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            base = layer_dir.name
            qweight_path = layer_dir / f"{bw}.npy"
            qparams_path = layer_dir / f"{bw}_qparams.npz"
            if not qweight_path.exists() or not qparams_path.exists():
                continue
            qweight = np.load(qweight_path)
            qp = np.load(qparams_path)
            scales = qp["scales"]
            zeros = qp["zeros"]

            # Load BF16 weight from source shards (BF16 shard keys have `model.` prefix)
            shard_key = f"model.{base}.weight" if not base.startswith("model.") else f"{base}.weight"
            if shard_key not in shard_map:
                continue
            with safe_open(str(shard_map[shard_key]), framework="pt", device="cpu") as f:
                w = f.get_tensor(shard_key)
            if hasattr(w, "detach"):
                w = np.asarray(w.detach().cpu().to(torch.float32))
            else:
                w = np.asarray(w).astype(np.float32)

            param_count = int(np.prod(w.shape))
            if base not in db["param_counts"]:
                db["param_counts"][base] = param_count

            # Dequantize: qweight is [output_dim, input_dim] (unpacked uint8)
            # scales/zeros are [output_dim, num_groups]; expand to full input dim
            repeat = w.shape[1] // scales.shape[1]
            if repeat > 1:
                s_exp = np.repeat(scales, repeat, axis=1)
                z_exp = np.repeat(zeros, repeat, axis=1)
            else:
                s_exp = scales
                z_exp = zeros
            wq = (qweight.astype(np.float32) * s_exp + z_exp)

            hessian = hessian_map.get(base)
            h = np.asarray(hessian).astype(np.float32) if hessian is not None else np.ones(w.shape[1])
            diff = w - wq
            err = float(np.sum(diff * diff * h) / (np.sum(w * w * h) + 1e-12))

            if base not in db["layers"]:
                db["layers"][base] = {}
            db["layers"][base][str(bw)] = {"error": err}
            all_errors += 1

    output.write_text(json.dumps(db, indent=2))
    print(f"\nDone. Wrote {output}")
    print(f"  Layers with errors: {len(db['layers'])}")
    print(f"  Total measurements: {all_errors}")


if __name__ == "__main__":
    main()
