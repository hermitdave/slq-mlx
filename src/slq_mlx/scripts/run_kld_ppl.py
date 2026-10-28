#!/usr/bin/env python3
"""Compute KLD and PPL between BF16 and SLQ checkpoint, caching logits to avoid loading both models at once."""

import argparse
import logging
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from slq_mlx.mlx_data import prepare_calibration_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_logits(model, input_ids, cache_path=None):
    """Compute logits for all sequences, optionally caching to disk."""
    if cache_path and Path(cache_path).exists():
        logger.info(f"Loading cached logits from {cache_path}")
        return np.load(cache_path)
    
    logger.info(f"Computing logits for {input_ids.shape[0]} sequences...")
    all_logits = []
    
    for i in range(input_ids.shape[0]):
        ids = input_ids[i:i+1]
        logits = model(ids)[:, :-1, :]
        # Convert bfloat16 to float32 for numpy compatibility
        all_logits.append(np.array(logits.astype(mx.float32)))
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{input_ids.shape[0]} sequences")
            mx.clear_cache()
    
    logits_array = np.concatenate(all_logits, axis=0)
    
    if cache_path:
        np.save(cache_path, logits_array)
        logger.info(f"Cached logits to {cache_path}")
    
    return logits_array


def compute_metrics_from_logits(logits_a, logits_b, input_ids):
    """Compute PPL and symmetric KLD from pre-computed logits."""
    all_ppl_a = []
    all_ppl_b = []
    all_kld = []

    for i in range(input_ids.shape[0]):
        logits_a_i = mx.array(logits_a[i:i+1])
        logits_b_i = mx.array(logits_b[i:i+1])
        
        logits_a_i = logits_a_i.astype(mx.float32)
        logits_b_i = logits_b_i.astype(mx.float32)

        targets = input_ids[i:i+1, 1:]

        logp_a = mx.log(mx.softmax(logits_a_i, axis=-1))
        logp_b = mx.log(mx.softmax(logits_b_i, axis=-1))

        p_a = mx.take_along_axis(logp_a, targets[..., None], axis=-1).squeeze(-1)
        p_b = mx.take_along_axis(logp_b, targets[..., None], axis=-1).squeeze(-1)

        # Compute KL in smaller chunks to avoid GPU timeout
        seq_len = p_a.shape[1]
        chunk_size = 64
        kl_ab_total = 0.0
        kl_ba_total = 0.0
        n_tokens = 0
        
        for j in range(0, seq_len, chunk_size):
            end = min(j + chunk_size, seq_len)
            p_a_chunk = p_a[:, j:end]
            p_b_chunk = p_b[:, j:end]
            
            exp_p_a = mx.exp(p_a_chunk)
            exp_p_b = mx.exp(p_b_chunk)
            mx.eval(exp_p_a, exp_p_b)
            
            kl_ab_chunk = mx.sum(exp_p_a * (p_a_chunk - p_b_chunk), axis=-1)
            kl_ba_chunk = mx.sum(exp_p_b * (p_b_chunk - p_a_chunk), axis=-1)
            mx.eval(kl_ab_chunk, kl_ba_chunk)
            
            kl_ab_total += float(mx.sum(kl_ab_chunk))
            kl_ba_total += float(mx.sum(kl_ba_chunk))
            n_tokens += kl_ab_chunk.shape[0]
            
            mx.clear_cache()
        
        sym_kl = (kl_ab_total + kl_ba_total) / (2 * n_tokens)

        nll_a = -float(mx.mean(p_a))
        nll_b = -float(mx.mean(p_b))

        all_ppl_a.append(float(mx.exp(nll_a)))
        all_ppl_b.append(float(mx.exp(nll_b)))
        all_kld.append(sym_kl)

        mx.clear_cache()

        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i+1}/{input_ids.shape[0]} sequences")

    results = {
        "ppl_a": float(np.mean(all_ppl_a)),
        "ppl_b": float(np.mean(all_ppl_b)),
        "kld_sym": float(np.mean(all_kld)),
        "n_sequences": len(all_ppl_a),
    }
    return results


def main():
    ap = argparse.ArgumentParser(description="KLD/PPL validation: BF16 vs SLQ")
    ap.add_argument("--model-a", required=True, help="BF16 model path")
    ap.add_argument("--model-b", required=True, help="SLQ model path")
    ap.add_argument("--calibration-samples", type=int, default=512)
    ap.add_argument("--calibration-seq-length", type=int, default=512)
    ap.add_argument("--calibration-data", default="fineweb_edu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--lazy", action="store_true")
    ap.add_argument("--logits-cache-dir", default=None, help="Directory to cache logits")
    args = ap.parse_args()

    logger.info("Loading calibration data...")
    input_ids, _ = prepare_calibration_data(
        args.model_a,
        args.calibration_data,
        args.calibration_samples,
        args.calibration_seq_length,
        args.seed,
    )

    # Load model A and compute logits
    logger.info(f"Loading model A: {args.model_a}")
    model_a, _ = load(args.model_a, trust_remote_code=args.trust_remote_code, lazy=args.lazy)
    
    cache_a = None
    if args.logits_cache_dir:
        cache_a = Path(args.logits_cache_dir) / f"logits_a_{args.calibration_samples}_{args.calibration_seq_length}.npy"
    
    logits_a = compute_logits(model_a, input_ids, cache_path=str(cache_a) if cache_a else None)
    del model_a
    mx.clear_cache()
    
    # Load model B and compute logits
    logger.info(f"Loading model B: {args.model_b}")
    model_b, _ = load(args.model_b, trust_remote_code=args.trust_remote_code, lazy=args.lazy)
    
    cache_b = None
    if args.logits_cache_dir:
        cache_b = Path(args.logits_cache_dir) / f"logits_b_{args.calibration_samples}_{args.calibration_seq_length}.npy"
    
    logits_b = compute_logits(model_b, input_ids, cache_path=str(cache_b) if cache_b else None)
    del model_b
    mx.clear_cache()

    # Compute metrics from cached logits
    logger.info("Computing metrics from cached logits...")
    results = compute_metrics_from_logits(logits_a, logits_b, input_ids)

    print()
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Model A (BF16)   PPL: {results['ppl_a']:.4f}")
    print(f"Model B (SLQ)    PPL: {results['ppl_b']:.4f}")
    print(f"PPL ratio (B/A): {results['ppl_b']/results['ppl_a']:.6f}")
    print(f"Symmetric KLD:   {results['kld_sym']:.6f}")
    print(f"Sequences:       {results['n_sequences']}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
