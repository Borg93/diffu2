#!/usr/bin/env bash
# Low-resource: where diffu synth SHOULD help most — scarce real data + synth. This is diffu's actual use
# case (Swedish archives with little labeled data). Baseline 200-real-only vs 200-real + {800, 2000} clean
# synth (mix). With real scarce, synth has room to add signal instead of just diluting a strong baseline.
#
#   GPU=2 scripts/exp_htr_lowres.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; EPOCHS=${EPOCHS:-10}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

head -n 200 "$OUT/real/train.jsonl" > "$OUT/_real200.jsonl"

log "baseline: 200 real only ($EPOCHS epochs)"
uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/_real200.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_r200" --epochs "$EPOCHS" --track

for n in 800 2000; do
  head -n "$n" "$OUT/synth_fixed/train.jsonl" > "$OUT/_s${n}.jsonl"
  cat "$OUT/_real200.jsonl" "$OUT/_s${n}.jsonl" > "$OUT/_r200_s${n}.jsonl"
  log "200 real + $n synth = $(wc -l < "$OUT/_r200_s${n}.jsonl") lines"
  uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/_r200_s${n}.jsonl" --val "$OUT/test/test.jsonl" \
    --out "$OUT/htr_r200_s${n}" --epochs "$EPOCHS" --track
done

{ echo "=== LOW-RESOURCE: 200 real ± clean synth — CER on 1000 real IAM test lines ==="
  echo "-- 200 real only:"; uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_r200"
  for n in 800 2000; do
    echo "-- 200 real + $n synth:"; uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_r200_s${n}"
  done
} 2>&1 | tee "$OUT/results_lowres.txt"
log "low-resource done -> $OUT/results_lowres.txt (synth helps iff 200+synth CER < 200-only)"
