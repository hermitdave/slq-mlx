#!/usr/bin/env python3
"""DWQ refinement for official LFM3-1.7B 4-bit quant, keeping BF16 teacher loaded."""

import argparse
import time
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
import numpy as np
from tqdm import tqdm

from mlx_lm.utils import load, load_tokenizer, save
from mlx_lm.quant.dwq import dwq_quantize
from mlx_lm.tuner.trainer import iterate_batches
from mlx_lm.tuner.losses import kl_div_loss


def load_calibration(cache_path, num_samples, max_seq_length):
    """Load cached calibration data."""
    data = np.load(cache_path, allow_pickle=True)
    input_ids = data['input_ids']
    
    # Limit samples
    input_ids = input_ids[:num_samples]
    
    # Create samples with full length
    samples = []
    for tokens in input_ids:
        samples.append((tokens, len(tokens)))
    
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='BF16 teacher model')
    parser.add_argument('--quantized-model', required=True, help='Student quantized model')
    parser.add_argument('--mlx-path', required=True, help='Output path')
    parser.add_argument('--bits', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=64)
    parser.add_argument('--num-samples', type=int, default=256)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--calibration-cache', default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(f'Loading BF16 teacher: {args.model}')
    teacher, _, config = load(
        args.model,
        return_config=True,
        lazy=True,
        trust_remote_code=args.trust_remote_code,
    )
    teacher.eval()
    print('BF16 teacher loaded and set to eval')
    
    # Materialize teacher with a dummy forward pass to avoid timeout during DWQ
    print('Materializing BF16 teacher...')
    dummy = mx.array([[1, 2, 3, 4, 5]])
    _ = teacher(dummy)
    mx.eval(_)
    mx.clear_cache()
    print('BF16 teacher materialized')

    print(f'Loading quantized student: {args.quantized_model}')
    student, tokenizer, q_config = load(
        args.quantized_model,
        lazy=True,
        return_config=True,
        trust_remote_code=args.trust_remote_code,
    )
    print(f'Student loaded, quantization: {q_config.get("quantization", {}).get("mode")}')

    # Unfreeze scales/biases only
    def unfreeze(_, m):
        if (hasattr(m, 'bits') and hasattr(m, 'group_size')
                and m.mode == 'affine' and m.bits < 8):
            m.unfreeze(keys=['scales', 'biases'], recurse=False)

    student.apply_to_modules(unfreeze)

    # Load calibration data
    if args.calibration_cache and Path(args.calibration_cache).exists():
        print(f'Loading cached calibration: {args.calibration_cache}')
        train_data = load_calibration(
            args.calibration_cache,
            args.num_samples,
            args.max_seq_length,
        )
        valid_data = load_calibration(
            args.calibration_cache,
            32,
            args.max_seq_length,
        )
        # Simple split: last 32 for valid
        if len(train_data) > 32:
            valid_data = train_data[-32:]
            train_data = train_data[:-32]
    else:
        raise ValueError('calibration-cache is required')

    print(f'Train samples: {len(train_data)}, Valid samples: {len(valid_data)}')

    # Target function: live teacher
    def target_fn(batch, idx, split):
        logits = teacher(batch)
        idx_top = mx.argpartition(logits, kth=-1024, axis=-1)[..., -1024:]
        logits_top = mx.take_along_axis(logits, idx_top, axis=-1)
        return logits_top, idx_top

    # Optimizer
    opt = optimizers.Adam(learning_rate=args.learning_rate, bias_correction=True)

    # Run DWQ
    print('Starting DWQ...')
    dwq_quantize(
        student,
        target_fn,
        opt,
        train_data,
        valid_data,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        gradient_checkpoint=False,
    )

    # Save
    print(f'Saving to: {args.mlx_path}')
    save(args.mlx_path, args.quantized_model, student, tokenizer, q_config)
    print('Done!')


if __name__ == '__main__':
    main()
