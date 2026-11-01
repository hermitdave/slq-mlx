#!/usr/bin/env python3
"""Stage 1 streaming: layer-by-layer GPTQ with aggressive memory release.

For large BF16 models that barely fit in RAM (e.g. 27B at ~54 GB on 64 GB),
this script processes one transformer block at a time and releases each block's
BF16 weights before moving to the next. It reuses the existing MLXFastOBQ and
MLXQuantizer internals but wraps them with explicit cache clearing.

Usage:
    python3 -m slq_mlx.scripts.run_gptq_streaming \
        --model /path/to/BF16/source \
        --bitwidths 2 3 4 5 6 8 \
        --cal-bitwidth 4 \
        --save-dir ./quant/Qwen3.8-27B-Coder390/4bit \
        --calibration-tokens 65536 \
        --calibration-seq-length 1024 \
        --dtype float16 \
        --lazy \
        --group-size 64 \
        --trust-remote-code
"""
import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load as load_model
from slq_mlx.mlx_fast_obq import MLXFastOBQ
from slq_mlx.mlx_quantizer import MLXQuantizer, QuantizedLinear
from slq_mlx.mlx_data import prepare_calibration_data
from slq_mlx.mlx_model_utils import (
    detect_quantizable_modules,
    detect_pre_block_modules,
    get_model_layers,
    get_submodule,
    set_submodule_weight,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gptq_streaming")

SUPPORTED_BITS = (2, 3, 4, 5, 6, 8)


def parse_args():
    p = argparse.ArgumentParser(description="Streaming layer-by-layer GPTQ for large models")
    p.add_argument("--model", required=True, help="HF model ID or local BF16 path")
    p.add_argument("--quantizable-modules", default=None)
    p.add_argument("--pre-block-modules", nargs="+", default=None)
    p.add_argument("--block-modules", default="layers")
    p.add_argument("--calibration-data", default="fineweb_edu",
                   choices=["fineweb_edu", "wikitext2", "c4"])
    p.add_argument("--calibration-tokens", type=int, default=2**20)
    p.add_argument("--calibration-seq-length", type=int, default=1024)
    p.add_argument("--bitwidths", nargs="+", type=int, default=[2, 3, 4, 5, 6, 8])
    p.add_argument("--cal-bitwidth", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--sym", action="store_true")
    p.add_argument("--perchannel", action="store_true", default=True)
    p.add_argument("--rel-damp", type=float, default=1e-2)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-dir", required=True)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--lazy", action="store_true", help="Lazy-load model weights")
    p.add_argument("--max-blocks", type=int, default=None, help="Smoke-test limit")
    p.add_argument("--release-after-block", action="store_true", default=True,
                   help="Release BF16 weights after each block is quantized")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cal_bitwidth not in args.bitwidths:
        logger.warning(f"cal-bitwidth {args.cal_bitwidth} not in bitwidths; using {args.bitwidths[0]}")
        args.cal_bitwidth = args.bitwidths[0]

    unsupported = [b for b in args.bitwidths if b not in SUPPORTED_BITS]
    if unsupported:
        raise ValueError(f"Unsupported bitwidths {unsupported}; MLX supports {SUPPORTED_BITS}")

    mx.random.seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load model (lazy + smaller dtype to stay within 64 GB)
    # ------------------------------------------------------------------ #
    logger.info("Loading model %s (lazy=%s, dtype=%s)...", args.model, args.lazy, args.dtype)
    t0 = time.perf_counter()
    model, tokenizer = load_model(
        args.model,
        trust_remote_code=args.trust_remote_code,
        lazy=args.lazy,
    )
    if args.dtype == "float16":
        model.set_dtype(mx.float16)
    elif args.dtype == "bfloat16":
        model.set_dtype(mx.bfloat16)
    else:
        model.set_dtype(mx.float32)

    if not args.lazy:
        mx.eval(model.parameters())
    logger.info("Model loaded in %.1f s", time.perf_counter() - t0)

    # Auto-detect modules
    if args.quantizable_modules is None:
        args.quantizable_modules = detect_quantizable_modules(model, args.block_modules)
        logger.info("Auto-detected regex: %s", args.quantizable_modules)
    if args.pre_block_modules is None:
        args.pre_block_modules = detect_pre_block_modules(model)
        logger.info("Pre-block modules: %s", args.pre_block_modules)

    # ------------------------------------------------------------------ #
    # 2. Calibration data
    # ------------------------------------------------------------------ #
    seq_len = args.calibration_seq_length
    n_samples = max(1, args.calibration_tokens // seq_len)
    logger.info("Preparing calibration data: %d samples x %d tokens", n_samples, seq_len)
    input_ids, _ = prepare_calibration_data(
        args.model, args.calibration_data, n_samples, seq_len, args.seed
    )
    data_loader = [((ids[None, :],), {}) for ids in input_ids]

    # ------------------------------------------------------------------ #
    # 3. Block-by-block streaming quantization
    # ------------------------------------------------------------------ #
    # Resolve the module that actually owns the transformer layers.
    # Some HF wrappers expose layers via a property that returns a slice,
    # so we must mutate the underlying list in-place.
    if "language_model" in model:
        inner = model["language_model"]
    elif "model" in model:
        inner = model["model"]
    else:
        inner = model
    blocks = get_model_layers(inner, args.block_modules)
    # Ensure we have the live list, not a slice/property copy
    if hasattr(inner, "layers") and not isinstance(inner.layers, list):
        inner = inner
        blocks = inner.layers
    elif hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
        blocks = inner.layers
    total_blocks = len(blocks)
    if args.max_blocks:
        blocks = blocks[: args.max_blocks]
        total_blocks = args.max_blocks

    logger.info("Quantizing %d / %d blocks at bitwidths %s",
                len(blocks), total_blocks, args.bitwidths)

    # Disable KV cache during calibration
    orig_cache = getattr(inner, "use_cache", None)
    inner.use_cache = False

    # Stage A: collect inputs to first block
    class _Collector(nn.Module):
        def __init__(self, mod):
            super().__init__()
            object.__setattr__(self, "mod", mod)
            object.__setattr__(self, "args", [])
            object.__setattr__(self, "kwargs", [])

        def __call__(self, *a, **kw):
            print(f"[COLLECTOR] called with {len(a)} args, shapes={[x.shape if hasattr(x,'shape') else 'N/A' for x in a]}")
            self.args.append(a)
            self.kwargs.append(kw)
            mx.eval(a, kw)
            raise Exception("Interrupt")

        def __getattr__(self, n):
            return getattr(object.__getattribute__(self, "mod"), n)

    blocks[0] = _Collector(blocks[0])
    print(f"[STAGE_A] Replaced blocks[0] with _Collector, type={type(blocks[0]).__name__}")
    for sample_idx, inp in enumerate(data_loader):
        print(f"[STAGE_A] Running sample {sample_idx}...")
        try:
            inner(*inp[0], **inp[1])
            print(f"[STAGE_A] Sample {sample_idx} completed without exception")
        except Exception as e:
            print(f"[STAGE_A] Sample {sample_idx} exception: {type(e).__name__}: {e}")
        print(f"[STAGE_A] blocks[0].args count={len(blocks[0].args)}, kwargs count={len(blocks[0].kwargs)}")
    input_args, input_kwargs = blocks[0].args, blocks[0].kwargs
    print(f"[STAGE_A] Final: input_args count={len(input_args)}, input_kwargs count={len(input_kwargs)}")
    blocks[0] = blocks[0].mod  # restore
    mx.eval(input_args, input_kwargs)

    # Stage B: process each block
    obq_kwargs = dict(
        group_size=args.group_size,
        sym=args.sym,
        perchannel=args.perchannel,
        rel_damp=args.rel_damp,
    )

    for block_id, block in enumerate(blocks):
        label = f"block {block_id}/{len(blocks) - 1}"
        # Resume: load block to discover projection names, skip if all bitwidths saved
        layers = {}
        for name, sub in block.named_modules():
            if isinstance(sub, nn.Linear):
                import re
                if re.search(args.quantizable_modules, name):
                    layers[name] = sub
        expected_dirs = [
            save_dir / f"{bw}bit" / f"{args.block_modules}.{block_id}.{name}"
            for bw in args.bitwidths
            for name in layers
        ]
        if expected_dirs and all(d.exists() and any(d.iterdir()) for d in expected_dirs):
            logger.info("[%s] SKIP (all %d bitwidths already quantized)", label, len(args.bitwidths))
            # Still need to recompute inputs for next block using current weights
            new_args, new_kwargs = [], []
            for a, kw in zip(input_args, input_kwargs):
                out = block(*a, **kw)
                out = out[0] if isinstance(out, (tuple, list)) else out
                na = list(a)
                if na:
                    na[0] = out
                elif "hidden_states" in kw:
                    kw = dict(kw)
                    kw["hidden_states"] = out
                new_args.append(tuple(na))
                new_kwargs.append(kw)
            input_args, input_kwargs = new_args, new_kwargs
            continue
        tb = time.perf_counter()

        layers = {}
        for name, sub in block.named_modules():
            if isinstance(sub, nn.Linear):
                import re
                if re.search(args.quantizable_modules, name):
                    layers[name] = sub

        logger.info("[%s] Quantizable layers: %d", label, len(layers))

        handles = {
            name: MLXFastOBQ(layer, args.bitwidths, **obq_kwargs)
            for name, layer in layers.items()
        }

        # Hook each layer with a wrapper module that captures inputs.
        originals = {}
        for name, layer in layers.items():
            parent_path = ".".join(name.split(".")[:-1])
            child = name.split(".")[-1]
            parent = get_submodule(block, parent_path)
            originals[name] = getattr(parent, child)
            handle = handles[name]

            class _Hook(nn.Module):
                def __init__(self, layer, hook_fn, label):
                    super().__init__()
                    object.__setattr__(self, "layer", layer)
                    object.__setattr__(self, "hook_fn", hook_fn)
                    object.__setattr__(self, "label", label)
                    self.weight = layer.weight
                    self.bias = getattr(layer, "bias", None)
                    self._count = 0

                def __call__(self, x):
                    self._count += 1
                    try:
                        self.hook_fn(x)
                    except Exception as e:
                        print(f"[HOOK ERROR] {self.label}: {e}")
                        raise
                    return self.layer(x)

            hooked = _Hook(originals[name], handle.update, name)
            setattr(parent, child, hooked)

        logger.info("[%s] Calibrating %d samples...", label, len(input_args))
        for sample_idx, (inp_args, inp_kwargs) in enumerate(zip(input_args, input_kwargs)):
            block(*inp_args, **inp_kwargs)

        # Restore originals
        for name, layer in originals.items():
            parent_path = ".".join(name.split(".")[:-1])
            parent = get_submodule(block, parent_path)
            setattr(parent, name.split(".")[-1], layer)

        logger.info("[%s] Calibration done, evaluating Hessians...", label)
        t_eval_start = time.perf_counter()
        # Batch eval: collect all Hessian arrays, evaluate once
        all_h = [h.H for h in handles.values()]
        mx.eval(*all_h)
        del all_h
        t_eval_end = time.perf_counter()
        logger.info("[%s] Hessians evaluated in %.2fs, starting quantize...", label, t_eval_end - t_eval_start)

        # Quantize + save (async disk writes)
        from slq_mlx.mlx_weight_store import save_qparams
        save_tasks = []
        cal_name = cal_qw = cal_sc = cal_zz = cal_handle = None

        # Pre-create all save dirs before launching any threads
        for name, handle in handles.items():
            full_name = f"{args.block_modules}.{block_id}.{name}"
            for bits in args.bitwidths:
                layer_save = save_dir / f"{bits}bit" / full_name
                layer_save.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for name, handle in handles.items():
                full_name = f"{args.block_modules}.{block_id}.{name}"
                quantized = handle.quantize()
                for bits, (qw, sc, zz) in quantized.items():
                    layer_save = save_dir / f"{bits}bit" / full_name
                    save_tasks.append(pool.submit(save_qparams, str(layer_save), bits, qw, sc, zz, handle))
                    if bits == args.cal_bitwidth:
                        cal_name, cal_qw, cal_sc, cal_zz, cal_handle = name, qw, sc, zz, handle

            # Replace cal_bitwidth weights in model for next-block calibration
            if cal_name is None:
                raise RuntimeError("cal_bitwidth quantized output missing")
            sc_expanded = mx.repeat(cal_sc, cal_handle.group_size, axis=1)[:, :cal_handle.d_col]
            zz_expanded = mx.repeat(cal_zz, cal_handle.group_size, axis=1)[:, :cal_handle.d_col]
            fake_w = cal_qw.astype(mx.float32) * sc_expanded + zz_expanded
            set_submodule_weight(block, cal_name, fake_w)

            # Wait for all disk writes to finish before moving on
            for task in save_tasks:
                task.result()

        # Recompute inputs for next block
        new_args, new_kwargs = [], []
        for a, kw in zip(input_args, input_kwargs):
            out = block(*a, **kw)
            out = out[0] if isinstance(out, (tuple, list)) else out
            na = list(a)
            if na:
                na[0] = out
            elif "hidden_states" in kw:
                kw = dict(kw)
                kw["hidden_states"] = out
            new_args.append(tuple(na))
            new_kwargs.append(kw)
        input_args, input_kwargs = new_args, new_kwargs

        del handles, originals, layers
        mx.eval(mx.zeros(1))
        mx.clear_cache()

        logger.info("[%s] Done in %.1f s", label, time.perf_counter() - tb)

    if orig_cache is not None:
        inner.use_cache = orig_cache

    logger.info("All %d blocks quantized -> %s", len(blocks), save_dir)


if __name__ == "__main__":
    main()
