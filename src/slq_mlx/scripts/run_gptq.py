#!/usr/bin/env python3
"""Stage 1: GPTQ quantization at all bitwidths.

Usage:
    python -m slq_mlx.scripts.run_gptq --model Qwen/Qwen3-8B --bitwidths 2 3 4 5 6 7 8 --cal-bitwidth 4 --save-dir ./quant
"""

import argparse
import logging
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_lm import load as load_model
from slq_mlx.mlx_fast_obq import MLXFastOBQ
from slq_mlx.mlx_quantizer import MLXQuantizer
from slq_mlx.mlx_data import prepare_calibration_data
from slq_mlx.mlx_model_utils import detect_quantizable_modules, detect_pre_block_modules

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="One-shot parallel GPTQ at multiple bitwidths")
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--quantizable-modules", default=None,
                        help="Regex matching Linear layers to quantize. Auto-detected from model architecture if omitted.")
    parser.add_argument("--pre-block-modules", nargs="+", default=None,
                        help="Module names before transformer blocks, e.g. embed_tokens norm. Auto-detected if omitted.")
    parser.add_argument("--block-modules", default="layers")
    parser.add_argument("--calibration-data", default="fineweb_edu", choices=["fineweb_edu", "wikitext2", "c4"])
    parser.add_argument("--calibration-tokens", type=int, default=2**23)
    parser.add_argument("--calibration-seq-length", type=int, default=2048)
    parser.add_argument("--bitwidths", nargs="+", type=int, default=[2,3,4,5,6,7,8], help="Bitwidths to quantize to (default: 2 3 4 5 6 7 8)")
    parser.add_argument("--cal-bitwidth", type=int, default=4, help="Calibration bitwidth (must be in --bitwidths, default: 4)")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument("--perchannel", action="store_true", default=True)
    parser.add_argument("--rel-damp", type=float, default=1e-2)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", required=True, help="Output directory for quantized layers")

    args = parser.parse_args()

    if args.cal_bitwidth not in args.bitwidths:
        parser.error(f"Calibration bitwidth {args.cal_bitwidth} not in --bitwidths")

    mx.random.seed(args.seed)

    logger.info(f"Loading model {args.model}...")
    model, tokenizer = load_model(args.model)
    # Model is already on default device in MLX-LM; ensure params are materialized
    mx.eval(model.parameters())

    if args.quantizable_modules is None:
        args.quantizable_modules = detect_quantizable_modules(model, args.block_modules)
        logger.info(f"Auto-detected quantizable modules: {args.quantizable_modules}")

    if args.pre_block_modules is None:
        args.pre_block_modules = detect_pre_block_modules(model)
        logger.info(f"Auto-detected pre-block modules: {args.pre_block_modules}")

    logger.info("Preparing calibration data...")
    seq_len = args.calibration_seq_length or model.config.max_position_embeddings
    n_samples = args.calibration_tokens // seq_len
    input_ids, _ = prepare_calibration_data(args.model, args.calibration_data, n_samples, seq_len, args.seed)

    data_loader = [((ids[None, :],), {}) for ids in input_ids]

    obq_kwargs = dict(
        group_size=args.group_size,
        sym=args.sym,
        perchannel=args.perchannel,
        rel_damp=args.rel_damp,
    )

    save_dir = Path(args.save_dir) / Path(args.model).name / f"{args.cal_bitwidth}bit"
    quantizer = MLXQuantizer(
        model=model,
        data_loader=data_loader,
        quantizable_modules=args.quantizable_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        save_dir=str(save_dir),
        obq_kwargs=obq_kwargs,
    )

    logger.info(f"Quantizing at bitwidths: {args.bitwidths}")
    quantizer.quantize(args.bitwidths, args.cal_bitwidth)

    logger.info(f"Done! Quantized layers saved to {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())