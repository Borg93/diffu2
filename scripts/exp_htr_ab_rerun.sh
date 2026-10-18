#!/usr/bin/env bash
# Re-run the HTR augmentation A/B on WIDTH-FIXED synth (see the _WIDTH_BIAS 0.55->0.45 fix + synth.py).
# The first A/B's synth was width-broken (garbled looping) -> its -2.4-pt "synth hurts" result is void.
# Real/test are reused (unchanged); the real-only baseline (htr_real, 9.50%) is reused. Only the synth is
# regenerated with the fixed path and the real+synth arm retrained, then both evaluated on real IAM test.
#
#   GPU=2 scripts/exp_htr_ab_rerun.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

OUT=runs/htr_ab
BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
ADAPTER=${ADAPTER:-runs/iam_lora/final}
HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
N_SYNTH=${N_SYNTH:-4000}; EPOCHS=${EPOCHS:-5}; GPU=${GPU:-2}
log(){ echo "[$(date +%H:%M:%S)] $*"; }
export CUDA_VISIBLE_DEVICES=$GPU

[ -f "$OUT/htr_real/model.safetensors" ] || { echo "missing real baseline $OUT/htr_real"; exit 1; }

# 1) regenerate synth with the FIXED width path (biased natural_width + ink_crop)
log "regenerating $N_SYNTH synth lines (width-fixed) on GPU $GPU -> $OUT/synth_fixed"
uv run diffu-synth --ckpt "$BASE" --lora "$ADAPTER" --hf-dataset Teklia/IAM-line --split train \
  --n "$N_SYNTH" --out "$OUT/synth_fixed"

# 2) rebuild the mixed manifest (reuse the unchanged real lines)
cat "$OUT/real/train.jsonl" "$OUT/synth_fixed/train.jsonl" > "$OUT/real_plus_synth_fixed.jsonl"
log "mixed manifest: $(wc -l < "$OUT/real_plus_synth_fixed.jsonl") lines"

# 3) train TrOCR on real + fixed-synth (trackio run htr_realsynth_fixed)
log "training TrOCR(real+synth_fixed) — $EPOCHS epochs"
uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/real_plus_synth_fixed.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_realsynth_fixed" --epochs "$EPOCHS" --track

# 4) evaluate both on held-out real IAM test — does width-fixed synth help now?
{ echo "=== WIDTH-FIXED synth A/B — CER on real IAM test (real=2000 synth=$N_SYNTH, epochs=$EPOCHS) ==="
  echo "-- TrOCR(real) [baseline, reused]:";  uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_real"
  echo "-- TrOCR(real+synth_fixed):";         uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_realsynth_fixed"
} 2>&1 | tee "$OUT/results_fixed.txt"
log "done -> $OUT/results_fixed.txt  (synth helps iff realsynth_fixed CER < 9.50%)"
