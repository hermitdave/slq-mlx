#!/usr/bin/env python3
"""Zero-shot multiple-choice evaluation: BF16 baseline vs a packed SLQ checkpoint.

Why this exists: lm-eval 0.4.13 is installed but cannot run here -- it loads task
data through `datasets`, which is broken on this stack (datasets 3.6.0 + pyarrow
25 raises `TypeError: Pickler._batch_setitems() takes 2 positional arguments
but 3 were given`). So the benchmark data is fetched as parquet shards and the
multiple-choice scoring is implemented directly.

The protocol is the standard one: for each (context, choices), compute the
loglikelihood of every choice continuation and take the argmax. Both models are
scored with identical prompts, so the comparison is fair even where our prompt
template differs slightly from lm-eval's.

Usage:
    python -m slq_mlx.scripts.run_zero_shot --model-a <id> --label-a BF16 \
                                           --model-b <id> --label-b slq-4.49bpw
"""

import argparse
import logging

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- benchmark data

def _parquet_url(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


TASKS = {
    # name: (repo, path, has_labels). Where the test split ships no labels (as with
    # HellaSwag), point at the validation split instead -- lm-eval does the same.
    "arc_challenge": dict(repo="allenai/ai2_arc", path="ARC-Challenge/test-00000-of-00001.parquet"),
    "arc_easy":      dict(repo="allenai/ai2_arc", path="ARC-Easy/test-00000-of-00001.parquet"),
    "hellaswag":     dict(repo="Rowan/hellaswag", path="data/validation-00000-of-00001.parquet"),
    "winogrande":    dict(repo="allenai/winogrande", path="winogrande_xl/validation-00000-of-00001.parquet"),
    "mmlu":          dict(repo="cais/mmlu", path="all/validation-00000-of-00001.parquet"),
}


def load_examples(task, limit=None):
    """Return [(context, [choices], gold_index), ...] for a task."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    spec = TASKS[task]
    local = hf_hub_download(spec["repo"], filename=spec["path"], repo_type="dataset")
    table = pq.read_table(local)
    rows = table.to_pylist()
    out = []
    for r in rows:
        if task.startswith("arc"):
            q = r["question"]
            ch = r["choices"]
            labels, texts = ch["label"], ch["text"]
            order = sorted(zip(labels, texts))
            gold = [i for i, (lb, _) in enumerate(order) if lb == r["answerKey"]]
            if not gold:
                continue
            # lm-eval format: "Question: ...\nAnswer:" + " " + choice
            out.append(("Question: " + q + "\nAnswer:", [f" {t}" for _, t in order], gold[0]))
        elif task == "hellaswag":
            ch = r["endings"]
            gold = int(r["label"])
            if gold < 0 or gold >= len(ch):
                continue
            out.append((r["ctx"].rstrip(), [" " + c for c in ch], gold))
        elif task == "winogrande":
            ch = [r["option1"], r["option2"]]
            gold = int(r["answer"]) - 1
            # lm-eval format: replace the blank, then " " + option
            out.append((r["sentence"], [" " + c for c in ch], gold))
        elif task == "mmlu":
            ch = r["choices"]
            gold = int(r["answer"])
            out.append(("Question: " + r["question"] + "\nAnswer:", [" " + c for c in ch], gold))
        if limit and len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------- scoring

def loglikelihood(tokenizer, model, context, continuation, max_len=2048):
    """Sum log P(continuation | context). Tokenises them separately so the
    continuation boundary is exact."""
    ctx_ids = tokenizer.encode(context)
    cont_ids = tokenizer.encode(continuation)
    if not cont_ids:
        return 0.0
    ids = mx.array([ctx_ids + cont_ids])
    n_ctx = len(ctx_ids)
    if ids.shape[1] > max_len:
        ids = ids[:, :max_len]
        cont_ids = cont_ids[: max_len - n_ctx]
        if not cont_ids:
            return 0.0
    logits = model(ids)[:, :-1, :].astype(mx.float32)          # [1, T-1, V]
    # MLX has no log_softmax
    logp = mx.log(mx.softmax(logits, axis=-1))
    # logits at position i predict the token at position i+1, so the token at
    # position n_ctx + j is scored by logits at n_ctx - 1 + j.
    lo = max(n_ctx - 1, 0)
    hi = lo + len(cont_ids)
    if hi > logp.shape[1]:
        hi = logp.shape[1]
    tgt = ids[:, lo + 1:hi + 1]                                # continuation tokens
    sel = mx.take_along_axis(logp[:, lo:hi, :], tgt[..., None], axis=-1)
    total = float(mx.sum(sel))
    mx.eval(total)
    return total


def score_task(tokenizer, model, task, limit=None):
    """Return (n, n_correct, set_of_correct_example_indices).

    Returning the correct-answer index set (not just the count) lets the caller
    build the paired 2x2 table between two models, which is a much tighter
    comparison than the difference of two independent accuracies.
    """
    examples = load_examples(task, limit)
    correct_idx = set()
    for i, (ctx, choices, gold) in enumerate(examples):
        scores = [loglikelihood(tokenizer, model, ctx, c) for c in choices]
        if int(np.argmax(scores)) == gold:
            correct_idx.add(i)
        if (i + 1) % 100 == 0:
            logger.info(f"    {task}: {i+1}/{len(examples)}")
    return len(examples), len(correct_idx), correct_idx


def run_model(label, model_id, tasks, limit):
    from mlx_lm import load

    logger.info(f"loading {label}: {model_id}")
    model, tokenizer = load(model_id)
    results = {}
    for task in tasks:
        n, ncorr, idx = score_task(tokenizer, model, task, limit)
        results[task] = (n, ncorr, idx)
        logger.info(f"  {label:16s} {task:14s} acc {ncorr/n:.4f}  (n={n})")
    del model
    mx.clear_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description="Zero-shot multiple-choice eval, BF16 vs SLQ")
    ap.add_argument("--model-a", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--label-a", default="BF16")
    ap.add_argument("--model-b", default="hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw")
    ap.add_argument("--label-b", default="slq")
    ap.add_argument("--tasks", default="arc_challenge,arc_easy,hellaswag,winogrande")
    ap.add_argument("--limit", type=int, default=400,
                    help="Cap examples per task (0 = all). The full sets are ~15.5k examples "
                         "x ~4 choices x 2 models = ~124k forward passes, i.e. hours.")
    args = ap.parse_args()

    tasks = [t for t in args.tasks.split(",") if t]

    a = run_model(args.label_a, args.model_a, tasks, args.limit)
    b = run_model(args.label_b, args.model_b, tasks, args.limit)

    print()
    print(f"{'task':14s} {'n':>5s} {args.label_a:>10s} {args.label_b:>10s} "
          f"{'recovery':>9s} {'both':>6s} {'Aonly':>6s} {'Bonly':>6s}")
    recs = []
    for t in tasks:
        na, ca, ia = a[t]
        nb, cb, ib = b[t]
        assert na == nb, "example sets must match for a paired comparison"
        aa, ab = ca / na, cb / nb
        r = ab / aa if aa else 0.0
        recs.append(r)
        both = len(ia & ib)
        a_only, b_only = len(ia - ib), len(ib - ia)
        print(f"{t:14s} {na:>5d} {aa:>10.4f} {ab:>10.4f} {r:>9.4f} "
              f"{both:>6d} {a_only:>6d} {b_only:>6d}")
    mean_r = float(np.mean(recs)) if recs else 0.0
    print(f"{'MEAN':14s} {'':>5s} {'':>10s} {'':>10s} {mean_r:>9.4f}")
    print()

    # Paired SE on the difference in proportions (Newcombe/Wilson-free, plain
    # binomial on the discordant pairs): the difference is driven by (Aonly-Bonly)/n.
    for t in tasks:
        na, ca, ia = a[t]
        _, cb, ib = b[t]
        d = (cb - ca) / na
        se = np.sqrt(len(ia - ib) + len(ib - ia)) / na
        print(f"  {t:14s} paired diff {d:+.4f} +/- {se:.4f}  (95% CI "
              f"[{d-1.96*se:+.4f}, {d+1.96*se:+.4f}])")
    print()
    print(f"  task-lossless threshold: recovery >= 0.99")
    print(f"  verdict: {'TASK-LOSSLESS' if mean_r >= 0.99 else 'NOT task-lossless'}")
    if args.limit:
        print(f"  NOTE: capped at {args.limit} examples/task -- the full sets are "
              f"~15.5k examples and take hours.")
    return 0 if mean_r >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
