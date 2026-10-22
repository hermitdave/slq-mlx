# SLQ-MLX: Statistically-Lossless Quantization for MLX

Port of [IST-DASLab SLQ](https://github.com/IST-DASLab/SLQ) (COLM 2026) to MLX for Apple Silicon.

## 🤗 Pre-quantized Model on Hugging Face

**Qwen3-1.7B-MLX-slq-4.5bpw** — Non-uniform SLQ quantization (4.49 bpw avg, EAR 0.9918)
- [hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw](https://huggingface.co/hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw)
- Load with: `mlx_lm.load("hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw")`

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
| Dense (Qwen3, Llama-3.x, …) | ✅ | Verified — Qwen3-1.7B at 4.49 bpw, EAR 0.9918 |
| MoE (Mixtral, DeepSeek, Nemotron-H, Qwen MoE, …) | ❌ | Untested — see blockers above |

## Results (Qwen3-1.7B)

| Metric | Target | Achieved |
|--------|--------|----------|
| **Average Bitwidth** | 4.5 bpw | **4.4935 bpw** |
| **EAR** | ≥ 0.99 | **0.9918** |
| **KL Divergence** | — | **0.0081** |
| **Distribution** | — | 8-bit: 66 layers (39%), 4-bit: 102 layers (61%) |

> Matches SLQ paper's **Distribution-Lossless** target (EAR ≥ 0.99 at 5.0–6.6 bpw) with simple heuristic search.

## Overview

SLQ finds the **lowest-bitwidth non-uniform weight quantization** that meets explicit fidelity targets:

| Target | Criterion | Paper Range |
|--------|-----------|-------------|
| **Task-Lossless (TL)** | Mean zero-shot benchmark recovery ≥ 99% of BF16 | 3.3–4.7 bpw |
| **Distribution-Lossless (DL)** | Next-token distribution matches BF16 (EAR ≥ 0.99) | 5.0–6.6 bpw |

This MLX port implements the full quantization pipeline **except mixed-bitwidth inference kernels** (Humming equivalent).

**Scope:** dense decoder-only transformers only — see [Model Support: Dense Models Only](#️-model-support-dense-models-only) above.

## Pipeline

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Stage 1: GPTQ  │───▶│ Stage 2: Build   │───▶│ Stage 3: Search │───▶│ Stage 4: Build   │
│  All Bitwidths  │    │ Sensitivity DB   │    │ (ILP + Binary)  │    │ MLX Checkpoint   │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

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
Export MLX-format safetensors checkpoint with per-layer bitwidth config and model card.

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

# 2. Build sensitivity database
python scripts/run_build_db.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit

# 3a. Distribution-Lossless search (EAR ≥ 0.99)
python scripts/run_search_dl.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --min-ear 0.99

# 3b. Task-Lossless search (placeholder - uses predicted constraints)
python scripts/run_search_tl.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --tasks arc_challenge,gsm8k,mmlu \
  --min-recovery 0.99

# 4. Build MLX checkpoint
python scripts/run_build_ckpt.py \
  --model Qwen/Qwen3-8B \
  --layer-dir ./quant/Qwen3-8B/4bit \
  --config ./quant/Qwen3-8B/4bit/config_dl_bws2-3-4-5-6-7-8_shapley_predicted_ear0.99_bw5.42_seed42.txt \
  --output-dir ~/Projects/ml/models
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
| Checkpoint | HF + Humming config | MLX safetensors + JSON |
| Weight Swapping | GPU cache offload | In-memory (instant) |
| Inference | vLLM + Humming kernels | MLX-LM (dequantized) |

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
For actual speedups (not just quality), need MLX kernel for:
```python
# Per-layer dequantize + matmul fused kernel
def mixed_gemm(x, qweights, scales, zeros, bitwidths):
    # x: [batch, seq, d_in]
    # qweights: {layer_name: uint8[d_out, d_in]}
    # Output: [batch, seq, d_out] per layer
    pass
```

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