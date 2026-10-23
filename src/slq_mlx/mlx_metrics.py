"""Metrics for SLQ: perplexity, top-K KL, EAR.

Ported from SLQ src/metrics/evaluation.py
"""

from typing import List, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import trange


def _token_nll(logits: mx.array, labels: mx.array) -> mx.array:
    """Per-token cross-entropy, shape of labels."""
    # logits: [batch, seq, vocab], labels: [batch, seq]
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    # Cross entropy
    # MLX has no mx.log_softmax; use log(softmax(...)).
    log_probs = mx.log(mx.softmax(logits_flat, axis=-1))
    nll = -log_probs[mx.arange(labels_flat.shape[0]), labels_flat]
    return nll.reshape(labels.shape)


def compute_perplexity(model: nn.Module, data: List[mx.array], batch_size: int = 1) -> Tuple[float, float, List[float]]:
    """Mean perplexity, variance of per-sample PPL, and per-sample PPL."""
    nlls = []
    for i in trange(0, len(data), batch_size, desc="Computing perplexity", leave=False):
        batch = mx.stack(data[i:i+batch_size])
        logits = model(batch)[:, :-1].astype(mx.float32)
        labels = batch[:, 1:]
        nll = _token_nll(logits, labels).mean(axis=1)
        nlls.extend(nll.tolist())

    nlls = np.array(nlls)
    ppl = np.exp(nlls)
    return float(np.exp(np.mean(nlls))), float(np.var(ppl)), ppl.tolist()


def compute_baseline_topk(
    model: nn.Module,
    input_ids: mx.array,
    batch_size: int = 4,
    top_k: int = 10,
) -> Tuple[mx.array, mx.array]:
    """Top-K baseline logits and indices at every position.

    Returns:
        topk_vals: [N, seq_len-1, top_k]
        topk_idx:  [N, seq_len-1, top_k]
    """
    model.eval()
    vals, idx = [], []
    for i in range(0, input_ids.shape[0], batch_size):
        batch = input_ids[i:i+batch_size]
        logits = model(batch)[:, :-1].astype(mx.float32)
        # Use argsort to get top-k indices, then gather values
        indices = mx.argsort(logits, axis=-1)[:, :, -top_k:]
        values = mx.take_along_axis(logits, indices, axis=-1)
        vals.append(values)
        idx.append(indices)
    return mx.concatenate(vals, axis=0), mx.concatenate(idx, axis=0)


def _nll_kl_ear(
    model: nn.Module,
    data: mx.array,
    topk_vals: mx.array,
    topk_idx: mx.array,
    batch_size: int,
    masks: Optional[mx.array],
    with_nll: bool,
) -> Tuple[float, float, float, int]:
    """Token-mean NLL, top-K KL(p||q) and EAR."""
    model.eval()
    # Callers pass the baseline cache straight from np.load, so coerce to MLX
    # arrays here rather than relying on every caller to do it.
    if not isinstance(topk_vals, mx.array):
        topk_vals = mx.array(topk_vals)
    if not isinstance(topk_idx, mx.array):
        topk_idx = mx.array(topk_idx)
    nll = kl = ear = 0.0
    n = 0

    for i in range(0, data.shape[0], batch_size):
        batch = data[i:i+batch_size]
        logits = model(batch)[:, :-1].astype(mx.float32)

        w = masks[i:i+batch_size, 1:].astype(mx.float32) if masks is not None else mx.ones_like(batch[:, 1:], dtype=mx.float32)

        if with_nll:
            nll += float(mx.sum(_token_nll(logits, batch[:, 1:]) * w))

        # KL(p||q) and EAR on top-K
        log_p = mx.log(mx.softmax(topk_vals[i:i+batch_size], axis=-1))
        log_q = mx.log(mx.softmax(
            mx.take_along_axis(logits.astype(mx.float32), topk_idx[i:i+batch_size], axis=-1),
            axis=-1
        ))
        p = mx.exp(log_p)

        kl += float(mx.sum(mx.sum(p * (log_p - log_q), axis=-1) * w))
        ear += float(mx.sum(mx.sum(mx.minimum(p, mx.exp(log_q)), axis=-1) * w))

        n += float(mx.sum(w))

    n = max(n, 1.0)
    return nll / n, kl / n, ear / n, int(n)


def compute_all_metrics(
    model: nn.Module,
    calibration_data: mx.array,
    baseline_topk_vals: mx.array,
    baseline_topk_idx: mx.array,
    batch_size: int = 4,
    masks: Optional[mx.array] = None,
) -> Tuple[float, float, float]:
    """Mean NLL, top-K KL and EAR against baseline top-K cache."""
    return _nll_kl_ear(model, calibration_data, baseline_topk_vals, baseline_topk_idx, batch_size, masks, True)[:3]


def compute_kl_and_ear(
    model: nn.Module,
    baseline_topk_vals: mx.array,
    baseline_topk_idx: mx.array,
    eval_data: mx.array,
    batch_size: int = 4,
    masks: Optional[mx.array] = None,
) -> Tuple[float, float, int]:
    """(mean top-K KL, mean EAR, number of scored tokens)."""
    return _nll_kl_ear(model, eval_data, baseline_topk_vals, baseline_topk_idx, batch_size, masks, False)[1:]