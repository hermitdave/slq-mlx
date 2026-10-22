"""Calibration data loading for SLQ.

Ported from SLQ src/data/loading.py.

Note on dataset access: `datasets.load_dataset` is unusable in this environment
(datasets 3.6.0 alongside pyarrow 25 raises
``TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were
given`` on every load, streaming or not). The corpora are therefore fetched as
parquet shards with ``huggingface_hub`` and decoded with ``pyarrow`` instead.
"""

import logging
import random
from typing import List, Optional, Tuple
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MASKED_DATASETS = {"gsm8k", "openthoughts"}

# (repo, filename substring) for each supported calibration corpus.
CORPORA = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", ".parquet"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    "c4": ("allenai/c4", "en"),
}


def get_tokenizer(model_id: str):
    """Load tokenizer for model."""
    return AutoTokenizer.from_pretrained(model_id, use_fast=False)


def _parquet_shards(repo: str, pattern: str, seed: int) -> List[str]:
    """List a dataset repo's matching parquet shards in a reproducible order."""
    from huggingface_hub import list_repo_files

    files = [
        f for f in list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet") and pattern in f
    ]
    files.sort()
    random.Random(seed).shuffle(files)
    return files


def _parquet_text_sequences(
    repo: str,
    pattern: str,
    n_tokens: int,
    seq_len: int,
    tokenizer,
    seed: int = 42,
    max_shards: int = 6,
    batch_size: int = 512,
) -> List[mx.array]:
    """Tokenize a parquet text corpus and chunk it into fixed-length sequences."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    shards = _parquet_shards(repo, pattern, seed)
    if not shards:
        raise RuntimeError(f"No parquet shards matching {pattern!r} in {repo}")

    tokens: List[int] = []
    for shard in shards[:max_shards]:
        if len(tokens) >= n_tokens:
            break
        path = hf_hub_download(repo, shard, repo_type="dataset")
        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
        logger.info(f"  {shard}: {len(texts)} documents ({len(tokens)}/{n_tokens} tokens)")
        for i in range(0, len(texts), batch_size):
            if len(tokens) >= n_tokens:
                break
            for seq in tokenizer(texts[i:i + batch_size], add_special_tokens=False)["input_ids"]:
                tokens.extend(seq)
                if len(tokens) >= n_tokens:
                    break

    tokens = tokens[:n_tokens]
    n_seq = len(tokens) // seq_len
    if n_seq == 0:
        raise RuntimeError(
            f"Only {len(tokens)} tokens collected, need at least seq_len={seq_len}"
        )
    return [mx.array(tokens[i * seq_len:(i + 1) * seq_len]) for i in range(n_seq)]


def get_data(
    name: str,
    n_tokens: int,
    seq_len: int,
    tokenizer,
    train: bool = True,
    seed: int = 42,
) -> List[mx.array]:
    """Load a calibration corpus and tokenize it into sequences.

    Returns a list of input_ids tensors, each ``[seq_len]``.
    """
    if name not in CORPORA:
        raise ValueError(f"Unknown dataset: {name}")
    repo, pattern = CORPORA[name]
    return _parquet_text_sequences(repo, pattern, n_tokens, seq_len, tokenizer, seed)


def get_fineweb_edu_gptq(n_tokens: int, seq_len: int, tokenizer, seed: int = 42) -> List[mx.array]:
    """FineWeb-Edu calibration sequences (10BT sample, parquet shards)."""
    repo, pattern = CORPORA["fineweb_edu"]
    return _parquet_text_sequences(repo, pattern, n_tokens, seq_len, tokenizer, seed)


def get_simple_calibration(n_tokens: int, seq_len: int, tokenizer, seed: int = 42) -> List[mx.array]:
    """Fallback calibration from a small fixed corpus (no network access).

    Only used when the real corpora are unavailable. It is far too small and too
    repetitive to estimate a useful Hessian, so a run that falls back to it is
    not comparable to a FineWeb-Edu run.
    """
    texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning is a subset of artificial intelligence. ",
        "Python is a popular programming language for data science. ",
        "The transformer architecture revolutionized natural language processing. ",
        "Quantization reduces model size while preserving accuracy. ",
        "Apple Silicon provides unified memory for efficient inference. ",
        "GPTQ is a post-training quantization method for LLMs. ",
        "SLQ finds statistically-lossless quantization configurations. ",
    ]
    text = " ".join(texts * (n_tokens // 200 + 1))[:n_tokens * 4]
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0][:n_tokens]
    n_seq = len(tokens) // seq_len
    return [mx.array(tokens[i * seq_len:(i + 1) * seq_len]) for i in range(n_seq)]


def prepare_calibration_data(
    model_id: str,
    dataset: str,
    n_samples: int,
    seq_len: int,
    seed: int = 42,
) -> Tuple[mx.array, Optional[mx.array]]:
    """Prepare calibration data as a single tensor ``[n_samples, seq_len]``.

    Returns ``(input_ids, masks)`` where masks is None for non-masked datasets.

    ``dataset`` is honoured. The previous implementation ignored it and always
    built a tiny synthetic corpus, so a requested 1M-token FineWeb-Edu run
    actually calibrated on ~10K tokens of repeated sentences.
    """
    tokenizer = get_tokenizer(model_id)
    total = n_samples * seq_len

    try:
        sequences = get_data(dataset, total, seq_len, tokenizer, seed=seed)
    except Exception as exc:
        logger.warning(
            f"Could not load {dataset!r} ({type(exc).__name__}: {exc}); falling "
            "back to the synthetic corpus, which is far too small for a reliable "
            "Hessian"
        )
        sequences = get_simple_calibration(total, seq_len, tokenizer, seed)

    if len(sequences) < n_samples:
        logger.warning(
            f"Requested {n_samples} calibration sequences but got {len(sequences)} "
            f"({len(sequences) * seq_len} tokens)"
        )
    sequences = sequences[:n_samples]

    input_ids = mx.stack(sequences)
    masks = mx.ones_like(input_ids) if dataset in MASKED_DATASETS else None
    return input_ids, masks
