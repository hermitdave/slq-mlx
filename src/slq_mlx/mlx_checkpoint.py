"""Build MLX checkpoint from SLQ bitwidth config.

DEPRECATED as the default export path. This writes fake-quantized float16
weights, which is always BF16-sized regardless of the SLQ bitwidth.

Use ``slq_mlx.scripts.run_pack_mlx`` instead -- it stores the integer grid
packed and runs it through MLX's fused ``mx.quantized_matmul``, so the model
actually honours the SLQ bitwidth (2.29x smaller than BF16 on Qwen3-1.7B).
``slq_mlx.scripts.run_build_ckpt`` now defaults to the packed exporter and only
reaches this module via its ``--dequantized`` flag.

Ported from SLQ slq_build_checkpoint.py
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from collections import Counter

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors.mlx import save_file

from .mlx_weight_store import WeightStore
from .mlx_model_utils import get_submodule, set_submodule, set_submodule_weight
from .mlx_sensitivity import load_model
from .mlx_data import get_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Header keys copied into model card
CARD_KEYS = (
    "constraint_min_ear", "min_recovery", "final_kl", "final_ear", "calibrated_kl",
    "predicted_kl", "predicted_ear", "predicted_recovery"
)

# vLLM fused module mapping (for reference; not used in MLX)
VLLM_FUSED = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
}


def parse_config_file(config_path: Path) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Parse '# Key: value' header lines and 'layer: bits' lines."""
    config, metadata = {}, {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                key, sep, value = line[1:].partition(":")
                if sep:
                    metadata[key.strip().lower().replace(" ", "_")] = value.strip()
            elif ":" in line:
                name, bits = line.rsplit(":", 1)
                config[name.strip()] = int(bits)
    return config, metadata


def model_card(metadata: Dict[str, str], config: Dict[str, int], group_size: int, dtype: str) -> str:
    """Generate HF-style model card (README.md)."""
    base = metadata.get("model", "unknown")
    counts = ", ".join(f"{b}-bit: {n}" for b, n in sorted(Counter(config.values()).items()))
    search = "\n".join(f"- {k}: {metadata[k]}" for k in CARD_KEYS if k in metadata)

    bits = set(config.values())
    if len(bits) == 1:
        desc = f"Uniform {bits.pop()}-bit weight quantization (SLQ calibration anchor)"
    else:
        desc = "Statistically-lossless mixed-bitwidth weight quantization"

    run = f"""A standard MLX checkpoint:
```bash
mlx_lm.generate --model <this-dir> --dtype {dtype}
```"""

    return f"""---
library_name: mlx
pipeline_tag: text-generation
base_model: {base}
tags:
- slq
- quantization
---

# {base} quantized with SLQ

{desc} of [{base}](https://huggingface.co/{base}); weights are stored fake-quantized in {dtype} on the integer grid. See the [SLQ paper](https://arxiv.org/abs/2605.02404) and the [code](https://github.com/IST-DASLab/SLQ).

## Quantization summary

- Average bitwidth (parameter-weighted): **{metadata.get('average_bitwidth', 'unknown')}** bpp
- Asymmetric INT GPTQ, group size {group_size}
- Layers per bitwidth: {counts}
{search}

## How to run

{run}

`quantization_config.txt` is the SLQ search output (layer: bits).

## Citation

```bibtex
@inproceedings{{helcig2026slq,
  title = {{Statistically-Lossless Quantization of Large Language Models}},
  author = {{Michael Helcig and Eldar Kurtic and Dan Alistarh}},
  booktitle = {{Conference on Language Modeling (COLM)}},
  year = {{2026}},
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Build MLX checkpoint from SLQ bitwidth config")
    parser.add_argument("--model", required=True, help="base model name or path")
    parser.add_argument("--layer-dir", required=True, help="quantized layer store (mlx_quantizer.py output)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="per-layer bitwidth config (layer: bits) from search")
    source.add_argument("--uniform", type=int, metavar="BITS", help="every layer at BITS (e.g. TL anchor)")
    parser.add_argument("--output-dir", required=True, help="writes <output-dir>/<model>/<subdir>")
    parser.add_argument("--subdir-name", default=None, help="default: config stem without 'config_', or uniform<BITS>")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                        help="On-disk weight dtype. float16 is 2 bytes/param -- identical to BF16, so the "
                             "artifact is never larger than its BF16 base. bfloat16 cannot be serialised "
                             "by safetensors.mlx (no PEP 3118 buffer format), which is why the old default "
                             "silently doubled the artifact to float32.")
    parser.add_argument("--group-size", type=int, default=128, help="must match the layer store")
    parser.add_argument("--force", action="store_true", help="overwrite existing output directory")
    args = parser.parse_args()

    weight_store = WeightStore(args.layer_dir).load()

    if args.uniform is not None:
        config = {n: args.uniform for n in weight_store.get_layer_names()}
        metadata = {}
        subdir = args.subdir_name or f"uniform{args.uniform}"
    else:
        config, metadata = parse_config_file(Path(args.config))
        subdir = args.subdir_name or Path(args.config).stem.removeprefix("config_")

    if not config:
        raise ValueError(f"No layers to quantize (config {args.config}, layer dir {args.layer_dir})")

    metadata.setdefault("model", args.model)

    out = Path(args.output_dir).expanduser() / Path(args.model).name / subdir
    if out.exists():
        if not args.force:
            logger.error(f"{out} exists; pass --force to overwrite")
            return 1
        shutil.rmtree(out)

    # Load model and apply quantization
    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)

    # Compute average bitwidth if not in metadata
    if "average_bitwidth" not in metadata:
        modules = dict(model.named_modules())
        numel = {n: modules[n].weight.size for n in config if n in modules}
        if numel:
            avg_bw = sum(config[n] * k for n, k in numel.items()) / sum(numel.values())
            metadata["average_bitwidth"] = f"{avg_bw:.3f}"

    # Apply quantized weights (set weight parameter on original Linear layers)
    for name, bits in config.items():
        weight = weight_store.get_layer_weight(name, bits)
        set_submodule_weight(model, name, weight)

    logger.info(f"Saving to {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Save model weights (keep 'model.' prefix for compatibility with mlx_lm.load)
    def flatten_params(d, prefix=''):
        result = {}
        for k, v in d.items():
            new_key = f'{prefix}{k}' if prefix else k
            if isinstance(v, dict):
                result.update(flatten_params(v, new_key + '.'))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        result.update(flatten_params(item, f'{new_key}.{i}.'))
                    else:
                        result[f'{new_key}.{i}'] = item
            else:
                result[new_key] = v
        return result

    nested_params = model.parameters()
    flat_params = flatten_params(nested_params)

    # safetensors.mlx cannot serialise bfloat16, so weights are written as
    # float16: 2 bytes/param, exactly the BF16 footprint. Writing float32 here
    # (the previous behaviour) made the quantised model 2x its BF16 base.
    save_dtype = mx.float16 if args.dtype == "float16" else mx.float32
    state_dict = {
        k: (v.astype(save_dtype) if isinstance(v, mx.array) else v)
        for k, v in flat_params.items()
    }
    save_file(state_dict, str(out / "model.safetensors"))
    total_bytes = sum(int(v.nbytes) for v in state_dict.values())
    logger.info(f"Saved weights: {total_bytes / 2**30:.2f} GiB as {args.dtype}")

    # Save model config
    import json
    config_dict = model.args if hasattr(model, 'args') else {}
    if hasattr(config_dict, '__dict__'):
        config_dict = config_dict.__dict__
    with open(out / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save tokenizer
    tokenizer.save_pretrained(str(out))

    # Save config
    if args.config:
        shutil.copy2(args.config, out / "quantization_config.txt")
    else:
        (out / "quantization_config.txt").write_text("\n".join(f"{n}: {b}" for n, b in config.items()))

    # Save README
    (out / "README.md").write_text(model_card(metadata, config, args.group_size, args.dtype))

    logger.info(f"Wrote {out} ({len(config)} layers, avg bitwidth {metadata.get('average_bitwidth', 'unknown')})")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())