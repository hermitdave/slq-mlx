#!/usr/bin/env python3
"""Benchmark tokens-per-second (TPS) for a quantized MLX model."""

import argparse
import time

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path or HF repo id")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    mx.random.seed(args.seed)

    print(f"Loading model: {args.model}")
    model, tokenizer = load(args.model, trust_remote_code=args.trust_remote_code)

    sampler = make_sampler(temp=0.0)
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_len = len(prompt_tokens)
    print(f"Prompt length: {prompt_len} tokens")

    # Warmup
    print(f"Warmup ({args.warmup} tokens)...")
    generate(model, tokenizer, args.prompt, max_tokens=args.warmup, sampler=sampler)
    mx.eval()
    mx.clear_cache()

    # Timed run
    print(f"Benchmarking {args.max_tokens} tokens...")
    t0 = time.perf_counter()
    text = generate(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        verbose=False,
    )
    t1 = time.perf_counter()
    mx.eval()

    gen_tokens = len(tokenizer.encode(text)) - prompt_len
    elapsed = t1 - t0
    tps = gen_tokens / elapsed if elapsed > 0 else 0

    print(f"Generated: {gen_tokens} tokens in {elapsed:.2f}s")
    print(f"TPS: {tps:.1f}")
    print(f"Output: {text[:120]}")


if __name__ == "__main__":
    main()
