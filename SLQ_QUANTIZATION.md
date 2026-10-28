# SLQ Quantization — 4.5 bpw

This is a mixed-precision quantized variant of **Ornith-1.5-9B** produced by the **SLQ (Structured Layer-wise Quantization)** pipeline from the [RCO-MLX](https://github.com/hermitdave/rco-mlx) project.

## Quantization details

| Parameter | Value |
|-----------|-------|
| Target bits per weight | 4.50 bpw |
| Achieved effective bpw | 5.00 bpw |
| Group size | 64 |
| Quantized tensors | 248 |
| Calibration dataset | FineWeb-Edu (128 samples × 2048 tokens) |
| Mean relative error | 0.00622 |

## Bitwidth distribution

| Bitwidth | Tensor count |
|----------|--------------|
| 8-bit | 25 |
| 6-bit | 43 |
| 4-bit | 180 |

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("hermitdave/Ornith-1.5-9B-SLQ-4.5bpw")
text = generate(model, tokenizer, "Hello, my name is", max_tokens=100)
```

## Methodology

SLQ uses Hessian-weighted per-layer error estimation to assign bitwidths:

1. A small 8-bit RTN model is built for activation capture
2. Per-layer error is measured at candidate bitwidths using real-text calibration
3. A greedy allocator assigns bits to meet the target bpw budget
4. Outlier layers with high error are auto-protected

## Attribution

Quantized model produced by **Hermes Agent** (Nous Research).
