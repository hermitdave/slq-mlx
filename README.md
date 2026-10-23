# SLQ-MLX: Statistically-Lossless Quantization for MLX

Port of [IST-DASLab SLQ](https://github.com/IST-DASLab/SLQ) (COLM 2026) to MLX for Apple Silicon.

## 🤗 Pre-quantized Model on Hugging Face

**Qwen3-1.7B-MLX-slq-4.49bpw** — Non-uniform SLQ quantization, 4.494 bpw
- [hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw](https://huggingface.co/hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw)
- Load with: `mlx_lm.load("hermitdave/Qwen3-1.7B-MLX-slq-4.49bpw", trust_remote_code=True)`
- **1.40 GiB** against a 3.21 GiB BF16 base — **2.29× smaller**, packed (not dequantized)

## ⚠️ Model Support: Dense Models Only

**SLQ-MLX supports dense (non-MoE) transformer models only.** Mixture-of-Experts models are explicitly out of scope and untested.

### Why

The upstream SLQ method itself was evaluated on **dense models only**:

| Source | Finding |
|--------|---------|
| [SLQ paper (arXiv 2605.02404v1)](https://arxiv.org/abs/2605.02404) | Evaluates 4 dense models — Qwen3-8B, Qwen3-32B, Qwen3.5-27B, Llama-3.3-70B-Instruct. **Zero occurrences of "MoE", "expert", "mixture", or "router" in the full text.** |
| [IST-DASLab/SLQ repo](https://github.com/IST-DASLab/SLQ) | No MoE/expert-specific files, configs, or scripts |

Three concrete blockers if you tried to force a MoE model through:

1. **Expert naming.** The upstream layer regex is `.*layers.*((q|k|v|o|gate|up|down)_proj)$`. This matches Qwen-style experts (which reuse `gate_proj`/`up_proj`/`down_proj`) but **misses Mixtral-style experts** (`w1`/`w2`/`w3`).
2. **Allocation granularity.** Upstream `src/models/grouping.py` keys groups on `(block_index, pattern_index)`, so **all experts in a block collapse into one group and share a single bitwidth** — no per-expert allocation.
3. **Group size divisibility.** Group size 128 requires `in_features % 128 == 0`; expert intermediate dimensions frequently violate this.

> **Note:** the upstream README references a "Table 6 (MoE)", but in arXiv v1 that table is token-divergence analysis, not MoE. Worth clarifying with the authors.

### Support matrix

| Model type | Supported | Notes |
|------------|-----------|-------|
| Dense (Qwen3, Llama-3.x, …) | ✅ | Verified — Qwen3-1.7B at 4.494 bpw, 1.40 GiB packed artifact |
| MoE (Mixtral, DeepSeek, Nemotron-H, Qwen MoE, …) | ❌ | Untested — see blockers above |

## Results (Qwen3-1.7B)

Full 2–8 sweep, all 28 blocks GPTQ-calibrated on 1,048,576 FineWeb-Edu tokens.

| Metric | Value |
|--------|-------|
| **Average bitwidth** (parameter-weighted) | **4.4940 bpw** |
| **Distribution** | 6-bit 28.6%, 5-bit 37.8%, 4-bit 33.7% |
| **EAR** (FineWeb-Edu) | **0.9116** |
| **KL divergence** (FineWeb-Edu) | **0.0522** |
| **Artifact size** | **1.399 GiB** (BF16 base 3.205 GiB → 2.29×) |

Re-running the default heuristic path reproduces the same allocation (598 greedy steps, 6/5/4-bit split
identical) and validates at EAR 0.9099 / KL 0.0514 — the small EAR delta versus 0.9116 is the
validation path: `run_simple_search.py` measures through `QuantizedLinear` (integer grid) while the
default `run_search_dl.py` path measures through the dequantized weight. Both are valid readings of the
same allocation.

> **EAR 0.91 at 4.5 bpw is consistent with the paper, not a defect.** SLQ's
> Distribution-Lossless target (EAR ≥ 0.99) is defined at **5.0–6.6 bpw**;
> 3.3–4.7 bpw is the Task-Lossless range, which does not promise EAR ≥ 0.99.
> Always state which regime a number belongs to.

## Overview

SLQ finds the **lowest-bitwidth non-uniform weight quantization** that meets explicit fidelity targets:

| Target | Criterion | Paper Range |
|--------|-----------|-------------|
| **Task-Lossless (TL)** | Mean zero-shot benchmark recovery ≥ 99% of BF16 | 3.3–4.7 bpw |
| **Distribution-Lossless (DL)** | Next-token distribution matches BF16 (EAR ≥ 0.99) | 5.0–6.6 bpw |

**Scope:** dense decoder-only transformers only — see [Model Support: Dense Models Only](#️-model-support-dense-models-only) above.

## Why the artifact is actually smaller

The default export packs the integer grid and runs it through MLX's **fused**
quantized GEMM: `nn.QuantizedLinear.__call__` calls `mx.quantized_matmul`, which
dequantizes inside the kernel. The model therefore stays packed in memory, not
just on disk.

```
packed grid  ──► mx.quantized_matmul ──► x @ W  (dequantize fused in-kernel)
```

The older `run_build_ckpt --dequantized` path writes fake-quantized float16
weights instead. That artifact is always BF16-sized regardless of the SLQ
bitwidth, so it exists only for debugging or for runtimes without a quantized
GEMM. Measured on Qwen3-1.7B at 4.494 bpw:

| Artifact | Size | Notes |
|---|---|---|
| packed (default) | **1.399 GiB** | honours the 4.494 bpw; 2.29× smaller than BF16 |
| fake-quantized float16 | 3.205 GiB | same as BF16 — the bitwidth is discarded |

Why not 4×: the embedding (311M of 1,720M params, 18%) stays BF16, `mx.quantize`
only accepts bits ∈ {2,3,4,5,6,8} so 7-bit rounds up to 8, and per-group
scales/zeros add ~10%.

## Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Stage 1: GPTQ  │───▶│ Stage 2: Build   │───▶│ Stage 3: Search │───▶│ Stage 4: Export  │
│  All Bitwidths  │    │ Sensitivity DB   │    │ heuristic/ILP   │    │ Packed Checkpoint│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

### Which allocator to use

**Default: `--method heuristic`** — per-layer greedy allocation over `error_db.json`. Cheap, needs no
sensitivity database, and measured better than the ILP at equal budget on MLX.

**Optional: `--method ilp`** — multiple-choice knapsack over a Shapley or linear sensitivity database.
Requires stage 2. See the warning below: on MLX the ILP is normally *worse*, not better.

### Default grouping: upstream's spec (112 groups)

`--layer-groups` defaults to upstream's spec:

```
mlp.gate_proj,mlp.up_proj  self_attn.k_proj,self_attn.q_proj,self_attn.v_proj
```

Groups are formed **per transformer block**, and anything matching no pattern stays a singleton — so on a
28-layer model this yields **112 groups** (2 fused + 2 singletons per block), not 2. Grouping globally
across all blocks would collapse the model into 2 groups and leave the sensitivity DB with only 2 distinct
cost profiles, which is what makes the ILP nearly helpless.

Pass `--layer-groups ''` for a genuinely per-layer allocation (196 groups). That is the MLX-native choice,
because **MLX does not fuse SwiGLU or QKV** — the grouping exists because vLLM does, so those projections
must share a bitwidth *at serving time*. The default matches upstream so numbers stay comparable.

### Why the ILP is not the default on MLX

Measured on Qwen3-1.7B, FineWeb-Edu 32x1024, equal ~4.5 bpw budget:

| Allocator | Choices | Avg bpw | EAR | KL |
|---|---|---|---|---|
| **Heuristic (default)** | 196 | 4.4940 | **0.9116** | **0.0522** |
| ILP + Shapley (112-group default) | 112 | 4.5000 | 0.9072 | 0.0544 |
| ILP + linear (112-group default) | 112 | 4.5000 | 0.9072 | 0.0544 |

The ILP was verified correct — exhaustive enumeration of all 49 grouped combinations confirms it found
the true optimum of its search space (21 of 49 feasible at 4.5 bpw). The solver is fine; the grouped
search space is too small.

`--disable-grouping` is a **no-op** and will not fix this: Shapley values are computed at group
granularity, so the sensitivity DB holds only as many distinct cost profiles as there were groups.
Ungrouping cannot manufacture information the DB does not have.

### Sensitivity cost scales with group count

Both estimators call the metric once per (group, level/permutation) step:

| Grouping | Groups | Shapley calls @50 perms | Wall time @10.5 s |
|---|---|---|---|
| Global (2 pattern-sets, all blocks) | 2 | 900 | ~2.6 h |
| **Upstream default (per-block)** | **112** | **33,900** | **~99 h** |
| Per-layer (`--layer-groups ''`) | 196 | 59,100 | ~172 h |

**Upstream's own default granularity is already ~99 h on a single M3 Max.** So `run_build_db.py` refuses
sensitivity estimation above 50 groups unless you pass `--allow-expensive-sensitivity`, and it prints the
predicted call count first. To build only `error_db.json` (all the heuristic needs) pass
`--skip-linear --skip-shapley`.

### EAR ≥ 0.99 is not reachable on this corpus

The paper claims distribution-lossless compression (EAR ≥ 0.99) at ~5-6 bpw. Measured on Qwen3-1.7B,
FineWeb-Edu 32x1024, top-k 10, all layers uniform:

| Config | KL | EAR |
|---|---|---|
| BF16 | 0.000000 | 1.000000 |
| all-8-bit | 0.001004 | **0.986348** |
| all-6-bit | 0.007244 | 0.966474 |
| all-5-bit | 0.026197 | 0.934538 |
| all-4-bit | 0.102166 | 0.871704 |

**EAR 0.99 is not reachable even at 8-bit**, so a DL bisect to `--min-ear 0.99` walks to max bitwidth and
reports `satisfies: False` — correctly. The EAR implementation is a line-for-line port of upstream
`src/metrics/evaluation.py`, so the gap is not a metric bug. Always state the measured ceiling next to
the paper's number rather than quietly retargeting `--min-ear`.

### Stage 1: GPTQ Quantization
Run GPTQ at bitwidths {2,3,4,5,6,7,8} for all quantizable layers. Saves fake-quantized weights and quantization parameters per layer.

### Stage 2: Sensitivity Database
- **Error Database**: Relative Frobenius error per layer per bitwidth
- **Linear Alphas**: Gaussian noise injection → metric degradation / t²
- **Shapley Values**: 50 permutations × layer groups × bitwidths (fast on unified memory)

### Stage 3: Bitwidth Allocation Search
- **Distribution-Lossless**: Binary search + ILP for minimum bitwidth meeting EAR ≥ 0.99
- **Task-Lossless**: (Placeholder) Benchmark-in-the-loop search via lm-eval

### Stage 4: Checkpoint Export
Export a **packed** MLX checkpoint: the integer grid is bit-packed and the decoder `Linear` layers become fused `nn.QuantizedLinear`, so the SLQ bitwidth is honoured in memory and on disk.

```bash
python -m slq_mlx.scripts.run_build_ckpt \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --config   ./quant/Qwen3-8B/4bit/config_..._bw4.49_seed42.txt \
  --output-dir ~/Projects/ml/models
```

Then:

```python
from mlx_lm import load, generate
model, tokenizer = load("~/Projects/ml/models/Qwen3-8B/<subdir>", trust_remote_code=True)
```

`trust_remote_code=True` is required: mlx-lm imports the generated
`slq_model.py` to build the architecture. Only load checkpoints you trust.

`--dequantized` writes the legacy fake-quantized float16 artifact instead
(always BF16-sized — debug/fallback only).

## Installation

```bash
cd ~/Projects/ml/slq-mlx
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Quantize model at all bitwidths (calibration at 4-bit)
python scripts/run_gptq.py \
  --model Qwen/Qwen3-8B \
  --bitwidths 2 3 4 5 6 7 8 \
  --cal-bitwidth 4 \
  --save-dir ./quant

# 2. Build the error database (built by run_build_db, or standalone below)
python scripts/run_build_db.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --skip-linear --skip-shapley          # error_db.json only

# 3. DEFAULT: per-layer greedy heuristic allocation at a fixed budget
python scripts/run_search_dl.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --target-bitwidth 4.5

# 3b. OPTIONAL: Shapley sensitivity database, then the ILP allocator
#     (needs the grouping flag -- see "Why the ILP is not the default on MLX")
python scripts/run_build_db.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --layer-groups "mlp.gate_proj,mlp.up_proj,mlp.down_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj,self_attn.o_proj"

python scripts/run_search_dl.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --method ilp --sensitivity-method shapley \
  --min-ear 0.99

# 3c. Task-Lossless search (placeholder - uses predicted constraints)
python scripts/run_search_tl.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --tasks arc_challenge,gsm8k,mmlu \
  --min-recovery 0.99

# 4. Build the packed checkpoint (default)
python -m slq_mlx.scripts.run_build_ckpt \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --config ./quant/Qwen3-8B/4bit/config_heuristic_target4.50_bws2-3-4-5-6-7-8_bw4.49_seed42.txt \
  --output-dir ~/Projects/ml/models

# 4b. Or the legacy fake-quantized float16 artifact (always BF16-sized)
python -m slq_mlx.scripts.run_build_ckpt ... --dequantized
```

Load a packed checkpoint:

```python
from mlx_lm import load
model, tokenizer = load("~/Projects/ml/models/Qwen3-8B/<subdir>", trust_remote_code=True)
```

## Output Structure

```
quant/
└── Qwen3-8B/
    └── 4bit/
        ├── model.layers.0.self_attn.q_proj/
        │   ├── 2.npy          # qweight (uint8)
        │   ├── 2_qparams.npz # scales, zeros, group_size, shape (binary)
        │   ├── 3.npy
        │   ├── 3_qparams.npz
        │   └── ...
        ├── error_db.json      # Per-layer relative error per bitwidth
        ├── baseline_*.npz     # Top-K baseline cache
        ├── linear_*.json      # Linear sensitivity alphas
        ├── shapley_*.json     # Shapley sensitivity values
        └── config_*.txt       # Search output (layer: bits)
```

## Key Differences from PyTorch SLQ

| Aspect | PyTorch SLQ | SLQ-MLX |
|--------|-------------|---------|
| Hardware | CUDA multi-GPU | Apple Silicon (unified memory) |
| Distributed | `torch.distributed` | Single-device |
| ILP Solver | OR-Tools SCIP | `pulp` + CBC |
| Checkpoint | HF + Humming config | MLX safetensors + packed grid |
| Weight Swapping | GPU cache offload | In-memory (instant) |
| Inference | vLLM + Humming kernels | MLX-LM + `mx.quantized_matmul` (fused) |

## Validation Targets

Reproduce paper Table 1 on Qwen3-8B / Llama-3.1-8B:
- **TL**: 3.3–4.7 bpw at ≥99% benchmark recovery
- **DL**: 5.0–6.6 bpw at EAR ≥0.99

## Architecture Notes

### Unified Memory Advantage
Shapley estimation requires 50 permutations × ~30 layer groups × 7 bitwidths = ~10,500 weight swaps. On CUDA this requires host↔device copies. On MLX unified memory, swapping is just pointer reassignment — **~10x faster**.

### ILP Solver
Uses `pulp` with CBC (pure Python, works on ARM). For ~30 groups × 7 bitwidths, solves in seconds.

### Group Size 128
Requires `in_features % 128 == 0`. Most models satisfy this. If not, pad input features (handled automatically in `MLXFastOBQ`).

## Extending

### Add Task-Lossless (lm-eval) Search
Integrate `lm-eval` in `mlx_search.py`'s `run_task_lossless`:
1. Build checkpoint at each search step
2. Run `lm_eval` via subprocess
3. Compute recovery vs BF16 baseline
4. Binary search on average bitwidth

### Mixed-Bitwidth Inference

Already solved — no custom kernel needed. MLX ships `mx.quantized_matmul`
(exposed as `nn.QuantizedLinear`), which dequantizes inside the GEMM. The
packed exporter writes `uint32` grids in exactly that layout, so a mixed-bitwidth
model runs natively and stays small in memory:

```python
# what nn.QuantizedLinear.__call__ does
x = mx.quantized_matmul(x, w, scales=scales, biases=biases,
                        transpose=True, group_size=128, bits=bits, mode="affine")
```

The one constraint is `mx.quantize`'s supported set: group_size ∈ {32, 64, 128}
and bits ∈ {2, 3, 4, 5, 6, 8}. A 7-bit layer rounds up to 8.

### MoE Support (not currently available)

MoE is out of scope for this port. Making it work would require:

1. **Per-expert layer selection** — a `--quantizable-modules` regex that matches the target MoE naming scheme (Qwen-style `gate_proj`/`up_proj`/`down_proj` works; Mixtral-style `w1`/`w2`/`w3` does not).
2. **Per-expert grouping** — upstream `build_layer_groups` keys on `(block_index, pattern_index)`, collapsing all experts in a block into one bitwidth. Per-expert allocation needs the group key extended to include the expert index.
3. **Group-size validation** — verify `in_features % 128 == 0` for every expert intermediate dimension before quantizing.
4. **Router exclusion** — the MoE router/gate should be kept at BF16 and excluded from the regex.

Untested. Contributions welcome, but expect the allocation-quality picture to differ from dense models since per-expert sensitivity is the main lever MoE has.

## License

Apache-2.0 (same as upstream SLQ)

## Citation

```bibtex
@inproceedings{helcig2026slq,
  title = {Statistically-Lossless Quantization of Large Language Models},
  author = {Michael Helcig and Eldar Kurtic and Dan Alistarh},
  booktitle = {Conference on Language Modeling (COLM)},
  year = {2026},
  url = {https://arxiv.org/abs/2605.02404},
}
```

## Acknowledgments

- Original SLQ implementation by IST-DASLab (Michael Helcig, Eldar Kurtic, Dan Alistarh)
- GPTQ implementation adapted from EvoPress
- Ported to MLX for Apple Silicon by **Hermes Agent** (Nous Research)
- Maintained by hermitdave (HuggingFace: hermitdave)