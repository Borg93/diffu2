#!/usr/bin/env bash
# Curriculum: pretrain TrOCR on synth, THEN fine-tune on real — the standard synth->real recipe (how TrOCR
# itself was made), vs the naive real+synth MIXING. Stage A (synth pretrain) is shared with the --lora
# variant; stage B here is FULL fine-tune. Compares to real-only (9.50%) and the best mix (0.25:1, 9.01%).
#
#   GPU=2 scripts/exp_htr_curriculum.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; EP_PRE=${EP_PRE:-5}; EP_FT=${EP_FT:-5}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# Stage A: pretrain TrOCR on 4000 synth (shared; skip if already done)
if [ ! -f "$OUT/htr_synth_pretrain/model.safetensors" ]; then
  log "Stage A: pretrain TrOCR on 4000 synth ($EP_PRE epochs)"
  uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/synth_fixed/train.jsonl" --val "$OUT/test/test.jsonl" \
    --out "$OUT/htr_synth_pretrain" --epochs "$EP_PRE" --track
fi

# Stage B (full): fine-tune the synth-pretrained model on 2000 real
log "Stage B: full fine-tune the synth-pretrained model on 2000 real ($EP_FT epochs)"
uv run diffu-htr-train --base "$OUT/htr_synth_pretrain" --train "$OUT/real/train.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_curriculum_full" --epochs "$EP_FT" --track

{ echo "=== CURRICULUM (synth-pretrain -> real full-FT) vs baselines ==="
  echo "  real-only 9.50% | best MIX (0.25:1) 9.01%"
  echo "-- synth-pretrain ONLY (read real test):"; uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_synth_pretrain"
  echo "-- curriculum full (synth->real):";        uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_curriculum_full"
} 2>&1 | tee "$OUT/results_curriculum.txt"
log "curriculum done -> $OUT/results_curriculum.txt"
