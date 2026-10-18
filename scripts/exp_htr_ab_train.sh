#!/usr/bin/env bash
# HTR A/B — train + eval half (data already materialized under runs/htr_ab by exp_htr_ab.sh).
# Fine-tunes two TrOCR recognizers IN PARALLEL (real on GPU 1, real+synth on GPU 2), each logging
# to trackio (project 'diffu-htr'), then evaluates both on held-out REAL IAM test -> CER.
#   synth helps iff  CER(real+synth) < CER(real).
#
#   scripts/exp_htr_ab_train.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

OUT=runs/htr_ab
HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}   # English HTR base (IAM is English)
EPOCHS=${EPOCHS:-5}
log(){ echo "[$(date +%H:%M:%S)] $*"; }

for f in "$OUT/real/train.jsonl" "$OUT/real_plus_synth.jsonl" "$OUT/test/test.jsonl"; do
  [ -f "$f" ] || { echo "missing $f — run scripts/exp_htr_ab.sh first"; exit 1; }
done

# --- two fine-tunes in parallel, one GPU each (stagger 45s so trackio.init doesn't race the db) ---
log "TrOCR(real) -> GPU 1"
CUDA_VISIBLE_DEVICES=1 uv run diffu-htr-train --base "$HTRBASE" \
  --train "$OUT/real/train.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_real" --epochs "$EPOCHS" --track > "$OUT/train_real.log" 2>&1 &
PID_REAL=$!
sleep 45
log "TrOCR(real+synth) -> GPU 2"
CUDA_VISIBLE_DEVICES=2 uv run diffu-htr-train --base "$HTRBASE" \
  --train "$OUT/real_plus_synth.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_realsynth" --epochs "$EPOCHS" --track > "$OUT/train_realsynth.log" 2>&1 &
PID_RS=$!

FAIL=0
wait $PID_REAL || { log "TrOCR(real) FAILED (see $OUT/train_real.log)"; FAIL=1; }
log "TrOCR(real) done"
wait $PID_RS   || { log "TrOCR(real+synth) FAILED (see $OUT/train_realsynth.log)"; FAIL=1; }
log "TrOCR(real+synth) done"
[ "$FAIL" -eq 0 ] || { log "a training arm failed — not evaluating"; exit 1; }

# --- final CER on held-out REAL IAM test ---
{ echo "=== CER on real IAM test (base=$HTRBASE, real=2000 synth=4000, epochs=$EPOCHS) ==="
  echo "-- TrOCR(real):";       CUDA_VISIBLE_DEVICES=1 uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_real"
  echo "-- TrOCR(real+synth):"; CUDA_VISIBLE_DEVICES=1 uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_realsynth"
} 2>&1 | tee "$OUT/results.txt"
log "done -> $OUT/results.txt"
