#!/bin/bash
# LFM2.5-2.6B SLQ pipeline chain (stages 1-4)
# Launched 2026-10-25. Explicit quantizable-module regex required for LFM2:
# its blocks use `conv`/`feed_forward`/`self_attn.out_proj`, none of which the
# auto-detector recognises (it falls back to `.*(_proj)$` and would skip all 90
# feed_forward layers = 1981.8M of 2434.79M quantizable params).
set -u
cd /Users/hermit.dave/Projects/ml/slq_mlx
export HF_HOME=/Users/hermit.dave/Projects/ml/hf_cache

MODEL=/Users/hermit.dave/Projects/ml/models/LFM2.5-2.6B-BF16
STORE=quant/LFM2.5-2.6B-BF16/4bit
MODELS_OUT=/Users/hermit.dave/Projects/ml/models
REGEX='self_attn\.(q_proj|k_proj|v_proj|out_proj)|conv\.(in_proj|out_proj)|feed_forward\.(w1|w3|w2)'
LOG=logs/lfm25_chain.log
STATUS=logs/lfm25_chain_status.txt

: > "$LOG"
echo "CHAIN START $(date)" >> "$STATUS"

stage() {
  local name="$1"; shift
  echo "=== STAGE $name :: $(date) ===" >> "$LOG"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "=== STAGE $name EXIT=$rc :: $(date) ===" >> "$LOG"
  echo "$name EXIT=$rc $(date)" >> "$STATUS"
  return $rc
}

# ---- Stage 1: GPTQ at all bitwidths --------------------------------------
# 16384 FineWeb-Edu tokens @ seq 2048 = 8 calibration samples.
stage 1_GPTQ python3 -u -m slq_mlx.scripts.run_gptq \
  --model "$MODEL" \
  --quantizable-modules "$REGEX" \
  --bitwidths 2 3 4 5 6 8 --cal-bitwidth 4 \
  --calibration-tokens 16384 --calibration-seq-length 2048 \
  --save-dir ./quant || exit 1

# ---- Stage 2: error database only ----------------------------------------
# 32x1024 = the validated apples-to-apples baseline from the Qwen3-1.7B run.
stage 2_ERROR_DB python3 -u -m slq_mlx.scripts.run_build_db \
  --model "$MODEL" --layer-dir "$STORE" \
  --skip-linear --skip-shapley \
  --calibration-samples 32 --calibration-seq-length 1024 || exit 1

# ---- Stage 3: per-layer greedy heuristic at 4.5 bpw ----------------------
stage 3_SEARCH python3 -u -m slq_mlx.scripts.run_search_dl \
  --model "$MODEL" --layer-dir "$STORE" \
  --method heuristic --target-bitwidth 4.5 \
  --calibration-samples 32 --calibration-seq-length 1024 \
  --top-k 10 --seed 42 || exit 1

# ---- Stage 4: packed checkpoint ------------------------------------------
CONFIG=$(ls -t "$STORE"/config_dl_*heuristic*target4.50*.txt 2>/dev/null | head -1)
echo "CONFIG=$CONFIG" >> "$STATUS"
if [ -z "$CONFIG" ]; then
  echo "NO CONFIG FOUND - stopping before stage 4" >> "$STATUS"
  exit 1
fi

stage 4_PACK python3 -u -m slq_mlx.scripts.run_build_ckpt \
  --model "$MODEL" --layer-dir "$STORE" \
  --config "$CONFIG" \
  --output-dir "$MODELS_OUT" --group-size 128 || exit 1

echo "CHAIN DONE $(date)" >> "$STATUS"
