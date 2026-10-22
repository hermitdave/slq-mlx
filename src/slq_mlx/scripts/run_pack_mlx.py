#!/usr/bin/env python3
"""Export an SLQ store as an MLX-native packed checkpoint.

This is the export path that honours the SLQ bitwidth. The decoder Linear
layers are replaced by MLX's fused ``nn.QuantizedLinear`` (``mx.quantized_matmul``
dequantizes inside the GEMM), so the model stays packed in memory as well as on
disk. mlx-lm loads it through the standard custom-architecture path::

    python -m slq_mlx.scripts.run_pack_mlx \\
        --model Qwen/Qwen3-1.7B \\
        --layer-dir ./quant/Qwen3-1.7B/4bit \\
        --config   ./quant/Qwen3-1.7B/4bit/config_..._bw4.49_seed42.txt \\
        --output-dir ~/Projects/ml/models

::

    from mlx_lm import load
    model, tokenizer = load(path, trust_remote_code=True)

``trust_remote_code=True`` is required because mlx-lm imports the generated
``slq_model.py`` to build the architecture.

For comparison, ``run_build_ckpt.py`` writes fake-quantized float16 weights.
That artifact is always BF16-sized regardless of the SLQ bitwidth and exists
only for debugging or for runtimes without a quantized GEMM.
"""

import argparse
import json
import logging
import sys

from slq_mlx.mlx_pack import SUPPORTED_GROUP_SIZES, pack_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Export an SLQ store as an MLX-native packed checkpoint"
    )
    parser.add_argument("--model", required=True, help="base model name or path")
    parser.add_argument("--layer-dir", required=True, help="quantized layer store")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="per-layer bitwidth config from the search")
    source.add_argument("--uniform", type=int, metavar="BITS",
                        help="every layer at BITS (calibration anchor)")
    parser.add_argument("--output-dir", required=True,
                        help="writes <output-dir>/<model>/<subdir>")
    parser.add_argument("--subdir-name", default=None)
    parser.add_argument("--group-size", type=int, default=128,
                        help=f"must match the store; one of {SUPPORTED_GROUP_SIZES}")
    parser.add_argument("--keep-dtype", default="float16",
                        choices=["float16", "float32"],
                        help="dtype for weights the search did not touch")
    parser.add_argument("--force", action="store_true", help="overwrite the output")
    args = parser.parse_args()

    if args.force:
        import shutil
        from pathlib import Path

        out = Path(args.output_dir) / Path(args.model).name
        if out.exists():
            shutil.rmtree(out)

    if args.uniform is not None:
        from slq_mlx.mlx_weight_store import WeightStore

        store = WeightStore(args.layer_dir).load()
        config = {n: args.uniform for n in store.get_layer_names()}
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for n, b in sorted(config.items()):
                fh.write(f"{n}: {b}\n")
            config_path = fh.name
    else:
        config_path = args.config

    summary = pack_checkpoint(
        model_id=args.model,
        layer_dir=args.layer_dir,
        config_path=config_path,
        output_dir=args.output_dir,
        group_size=args.group_size,
        keep_dtype=args.keep_dtype,
        subdir_name=args.subdir_name,
    )

    logger.info(f"Packed checkpoint : {summary['path']}")
    logger.info(f"Layers packed    : {summary['layers']}")
    logger.info(f"Quantized params : {summary['quantized_params']:,}")
    logger.info(f"Average bitwidth : {summary['average_bitwidth']:.4f}")
    logger.info(f"safetensors size  : {summary['safetensors_bytes'] / 2**30:.3f} GiB")
    if summary["rounded_up"]:
        logger.warning(f"{len(summary['rounded_up'])} layer(s) rounded up to a "
                       "supported bitwidth")

    return 0


if __name__ == "__main__":
    sys.exit(main())
