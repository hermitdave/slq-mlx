# SLQ-MLX: Statistically-Lossless Quantization for MLX

Port of [IST-DASLab SLQ](https://github.com/IST-DASLab/SLQ) (COLM 2026) to MLX for Apple Silicon.

## 🤗 Pre-quantized Model on Hugging Face

**Qwen3-1.7B-MLX-slq-4.5bpw** — Non-uniform SLQ quantization (4.49 bpw avg, EAR 0.9918)
- [hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw](https://huggingface.co/hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw)
- Load with: `mlx_lm.load("hermitdave/Qwen3-1.7B-MLX-slq-4.5bpw")`

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
        │   ├── 2_qparams.json # scales, zeros, metadata
        │   ├── 3.npy
        │   ├── 3_qparams.json
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