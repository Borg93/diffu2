#!/usr/bin/env bash
# Ratio sweep: does CLEAN (width-fixed) diffu synth HELP TrOCR on IAM at a lower synth:real ratio?
# real=2000 (fixed); clean rank-32 synth subset to {500,1000,2000} = ratios {0.25,0.5,1.0}:1. Already known:
# real-only baseline 9.50%, and 2:1 (4000 synth) 10.13%. Trains TrOCR per ratio, evals on held-out real
# IAM test, logs each to trackio (diffu-htr). synth helps at a ratio iff its CER < 9.50%.
#
#   GPU=2 scripts/exp_htr_ratio_sweep.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab
BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
ADAPTER=${ADAPTER:-runs/iam_lora_v2/final}
HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; EPOCHS=${EPOCHS:-5}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1) regenerate 4000 clean synth from the rank-32 adapter (fixed-width path) if missing
if [ ! -f "$OUT/synth_fixed/train.jsonl" ]; then
  log "regenerating 4000 clean synth from $ADAPTER on GPU $GPU"
  uv run diffu-synth --ckpt "$BASE" --lora "$ADAPTER" --hf-dataset Teklia/IAM-line --split train \
    --n 4000 --out "$OUT/synth_fixed"
fi
log "synth: $(wc -l < "$OUT/synth_fixed/train.jsonl") lines"

# 2) per-ratio train + eval
for pair in "0.25:500" "0.5:1000" "1.0:2000"; do
  ratio=${pair%%:*}; n=${pair##*:}
  m="$OUT/real_plus_synth_r${ratio}.jsonl"; hd="$OUT/htr_realsynth_r${ratio}"
  head -n "$n" "$OUT/synth_fixed/train.jsonl" > "$OUT/_synth_${n}.jsonl"
  cat "$OUT/real/train.jsonl" "$OUT/_synth_${n}.jsonl" > "$m"
  log "ratio ${ratio}:1  -> 2000 real + $n synth = $(wc -l < "$m") lines; training TrOCR"
  uv run diffu-htr-train --base "$HTRBASE" --train "$m" --val "$OUT/test/test.jsonl" \
    --out "$hd" --epochs "$EPOCHS" --track
done

# 3) eval all + summarize
{ echo "=== RATIO SWEEP: CER on 1000 real IAM test lines (real=2000, clean rank-32 synth) ==="
  echo "  reference: real-only 9.50% | 2:1 (4000 synth) 10.13%"
  for pair in "0.25:500" "0.5:1000" "1.0:2000"; do
    ratio=${pair%%:*}
    echo "-- ratio ${ratio}:1 --"; uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_realsynth_r${ratio}"
  done
} 2>&1 | tee "$OUT/results_ratio_sweep.txt"
log "ratio sweep done -> $OUT/results_ratio_sweep.txt (synth HELPS at a ratio iff CER < 9.50%)"
