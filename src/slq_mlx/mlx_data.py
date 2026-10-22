"""Calibration data loading for SLQ.

Ported from SLQ src/data/loading.py
"""

import os
from typing import List, Optional, Tuple
import mlx.core as mx
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


MASKED_DATASETS = {"gsm8k", "openthoughts"}


def get_tokenizer(model_id: str):
    """Load tokenizer for model."""
    return AutoTokenizer.from_pretrained(model_id, use_fast=False)


def get_data(
    name: str,
    n_tokens: int,
    seq_len: int,
    tokenizer,
    train: bool = True,
    seed: int = 42,
) -> List[mx.array]:
    """Load calibration dataset and tokenize to sequences.

    Returns list of input_ids tensors [seq_len] each.
    """
    if name == "fineweb_edu":
        return get_fineweb_edu_gptq(n_tokens, seq_len, tokenizer, seed)

    split = "train" if train else "test"
    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(ds["text"])
    elif name == "c4":
        ds = load_dataset("c4", "en", split=split, streaming=True)
        text = ""
        for i, ex in enumerate(ds):
            text += ex["text"] + " "
            if len(text) > n_tokens * 4:  # rough char estimate
                break
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Tokenize
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    tokens = tokens[:n_tokens]

    # Chunk into sequences
    n_seq = len(tokens) // seq_len
    sequences = []
    for i in range(n_seq):
        seq = tokens[i * seq_len:(i + 1) * seq_len]
        sequences.append(mx.array(seq))

    return sequences


def get_fineweb_edu_gptq(n_tokens: int, seq_len: int, tokenizer, seed: int = 42) -> List[mx.array]:
    """Load FineWeb-Edu dataset for GPTQ calibration."""
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    text = ""
    for ex in ds:
        text += ex["text"] + " "
        if len(text) > n_tokens * 4:
            break

    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    tokens = tokens[:n_tokens]

    n_seq = len(tokens) // seq_len
    sequences = []
    for i in range(n_seq):
        seq = tokens[i * seq_len:(i + 1) * seq_len]
        sequences.append(mx.array(seq))

    return sequences


def get_c4_streaming(n_tokens: int, seq_len: int, tokenizer, seed: int = 42) -> List[mx.array]:
    """Load C4 dataset using streaming (avoids fingerprinting issues)."""
    import random
    random.seed(seed)
    
    ds = load_dataset("c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    
    text = ""
    for ex in ds:
        text += ex["text"] + " "
        if len(text) > n_tokens * 4:
            break
    
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    tokens = tokens[:n_tokens]
    
    n_seq = len(tokens) // seq_len
    sequences = []
    for i in range(n_seq):
        seq = tokens[i * seq_len:(i + 1) * seq_len]
        sequences.append(mx.array(seq))
    
    return sequences


def get_simple_calibration(n_tokens: int, seq_len: int, tokenizer, seed: int = 42) -> List[mx.array]:
    """Simple calibration using a fixed text corpus (no external datasets)."""
    import random
    random.seed(seed)
    
    # Use a diverse set of texts for calibration
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Machine learning is a subset of artificial intelligence. " * 100,
        "Python is a popular programming language for data science. " * 100,
        "The transformer architecture revolutionized natural language processing. " * 100,
        "Quantization reduces model size while preserving accuracy. " * 100,
        "Apple Silicon provides unified memory for efficient inference. " * 100,
        "GPTQ is a post-training quantization method for LLMs. " * 100,
        "SLQ finds statistically-lossless quantization configurations. " * 100,
    ]
    
    # Repeat and shuffle
    text = " ".join(texts * (n_tokens // (len(" ".join(texts)) // 4) + 1))
    text = text[:n_tokens * 4]
    
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    tokens = tokens[:n_tokens]
    
    n_seq = len(tokens) // seq_len
    sequences = []
    for i in range(n_seq):
        seq = tokens[i * seq_len:(i + 1) * seq_len]
        sequences.append(mx.array(seq))
    
    return sequences


def prepare_calibration_data(
    model_id: str,
    dataset: str,
    n_samples: int,
    seq_len: int,
    seed: int = 42,
) -> Tuple[mx.array, Optional[mx.array]]:
    """Prepare calibration data as single tensor [n_samples, seq_len].

    Returns (input_ids, masks) where masks is None for non-masked datasets.
    """
    tokenizer = get_tokenizer(model_id)
    
    # Use simple calibration to avoid datasets library fingerprinting issues
    sequences = get_simple_calibration(n_samples * seq_len, seq_len, tokenizer, seed)
    
    # Take exactly n_samples
    sequences = sequences[:n_samples]
    
    # Stack to [n_samples, seq_len]
    input_ids = mx.stack(sequences)
    
    masks = None
    if dataset in MASKED_DATASETS:
        masks = mx.ones_like(input_ids)
    
    return input_ids, masks