#!/usr/bin/env bash
# Autonomous tail: WAIT for all generation chunks, POOL them, then run the two experiments in sequence —
# (1) the synth-AMOUNT curve, (2) the TARGETED-vs-UNIFORM payoff — all detached so the whole
# gen -> pool -> train -> eval chain finishes even if the interactive session drops. Idempotent: each stage
# is skipped if its results file already exists, so it is safe to relaunch. Launch:
#   NCHUNKS=20 nohup setsid bash scripts/exp_scaleup_orchestrate.sh > runs/scaleup_orchestrate.log 2>&1 </dev/null &
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; POOL="$OUT/synth100k/train.jsonl"; GPU=${GPU:-2}
NCHUNKS=${NCHUNKS:-20}; CHUNK=${CHUNK:-10000}
log(){ echo "[$(date '+%F %T')] $*"; }

log "orchestrator up — waiting for $NCHUNKS chunks x $CHUNK lines (all GPUs)"
while :; do
  ndone=0
  for c in $(seq 1 "$NCHUNKS"); do
    f="$OUT/synth100k/chunk_$c/train.jsonl"
    if [ -f "$f" ] && [ "$(wc -l <"$f" 2>/dev/null || echo 0)" -ge "$CHUNK" ]; then ndone=$((ndone+1)); fi
  done
  if [ "$ndone" -ge "$NCHUNKS" ]; then log "all $NCHUNKS chunks ready"; break; fi
  log "  $ndone/$NCHUNKS chunks done — waiting"; sleep 180
done

need=$(( NCHUNKS * CHUNK - 1000 ))
if [ ! -f "$POOL" ] || [ "$(wc -l <"$POOL")" -lt "$need" ]; then
  log "pooling -> $POOL"; cat "$OUT"/synth100k/chunk_*/train.jsonl > "$POOL"
fi
log "pool has $(wc -l <"$POOL") fakes"

if [ ! -f "$OUT/results_amount.txt" ]; then
  log "STAGE A: synth-amount curve"; GPU="$GPU" bash scripts/exp_htr_amount_sweep.sh
else log "STAGE A already done (results_amount.txt) — skip"; fi

if [ ! -f "$OUT/results_targeted.txt" ]; then
  log "STAGE B: targeted-vs-uniform payoff"; GPU="$GPU" bash scripts/exp_targeted_vs_uniform.sh
else log "STAGE B already done (results_targeted.txt) — skip"; fi

log "SCALE-UP COMPLETE"
echo "===== results_amount.txt ====="; cat "$OUT/results_amount.txt" 2>/dev/null || true
echo "===== results_targeted.txt ====="; cat "$OUT/results_targeted.txt" 2>/dev/null || true
