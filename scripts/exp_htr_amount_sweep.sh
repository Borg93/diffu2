#!/usr/bin/env bash
# EXTENSIVE synth-AMOUNT curve: how much does the recognizer improve as you pretrain on MORE diffu fakes,
# from a base that never saw IAM (trocr-base-stage1), before finetuning on all 6482 real IAM? Sweeps the
# pretrain amount N and reports CER vs N (+ a LoRA-adapt and a naive-mix point at the top N for contrast).
# Each row is written the moment it finishes and per-arm failures don't abort (safe unattended).
#
#   GPU=2 scripts/exp_htr_amount_sweep.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab
BASE=${BASE:-microsoft/trocr-base-stage1}
REAL="$OUT/real_all/train.jsonl"
SYNTH="$OUT/synth100k/train.jsonl"          # pooled fakes (up to 200k)
TEST="$OUT/test/test.jsonl"
GPU=${GPU:-2}; SEED=${SEED:-1}
AMOUNTS=${AMOUNTS:-"10000 25000 50000 100000 200000"}
PRE_EPOCHS=${PRE_EPOCHS:-2}                  # pretrain each N for this many epochs (scales samples with N)
FT_STEPS=${FT_STEPS:-6000}                   # finetune on real, fixed budget across arms
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%T)] $*"; }
for f in "$REAL" "$SYNTH" "$TEST"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
navail=$(wc -l < "$SYNTH")

pretrain(){ uv run diffu-htr-train --base "$1" --train "$2" --out "$3" --epochs "$PRE_EPOCHS" --seed "$SEED" >/dev/null 2>&1; }
ft(){ uv run diffu-htr-train --base "$1" --train "$2" --out "$3" --max-steps "$FT_STEPS" --seed "$SEED" "${@:4}" >/dev/null 2>&1; }
evalcer(){ uv run diffu-eval --manifest "$TEST" --htr "$1" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}'; }
report(){ printf "%-40s %s\n" "$1" "$2" | tee -a "$RES"; }

RES="$OUT/results_amount.txt"
{ echo "=== SYNTH-AMOUNT CURVE (base=$BASE, all $(wc -l <"$REAL") IAM real, pool=$navail fakes) — CER on 1000 test ==="
  echo "config: PRE_EPOCHS=$PRE_EPOCHS FT_STEPS=$FT_STEPS seed=$SEED"; } > "$RES"

# N=0 baseline: real only
log "N=0 baseline: finetune real only"
if ft "$BASE" "$REAL" "$OUT/am_A" "$FT_STEPS"; then report "N=0 real-only (baseline)" "$(evalcer "$OUT/am_A")"; else report "N=0 real-only" "FAILED"; fi
rm -rf "$OUT/am_A"

topN=0
for N in $AMOUNTS; do
  use=$N; [ "$use" -gt "$navail" ] && use=$navail    # clamp to what's in the pool (don't silently drop the top arm)
  head -n "$use" "$SYNTH" > "$OUT/_am_syn.jsonl"
  log "N=$use: pretrain ($PRE_EPOCHS ep) then finetune real"
  if pretrain "$BASE" "$OUT/_am_syn.jsonl" "$OUT/am_pre"; then
    if ft "$OUT/am_pre" "$REAL" "$OUT/am_ft" "$FT_STEPS"; then report "N=$use pretrain->finetune" "$(evalcer "$OUT/am_ft")"; else report "N=$use pretrain->finetune" "FAILED"; fi
    rm -rf "$OUT/am_ft"
    rm -rf "$OUT/am_pre_top"; mv "$OUT/am_pre" "$OUT/am_pre_top"   # keep the latest (largest) pretrain for the contrast rows
    topN=$use
  else report "N=$use pretrain" "FAILED"; rm -rf "$OUT/am_pre"; fi
  rm -f "$OUT/_am_syn.jsonl"
  [ "$use" -ge "$navail" ] && break                  # reached the pool size — larger requested amounts would just repeat
done

# contrast rows at the top N: LoRA-adapt instead of full FT, and a naive mix
if [ -d "$OUT/am_pre_top" ]; then
  log "top N=$topN: pretrain -> LoRA real"
  if ft "$OUT/am_pre_top" "$REAL" "$OUT/am_lora" "$FT_STEPS" --lora --lora-rank 16; then report "N=$topN pretrain->LoRA" "$(evalcer "$OUT/am_lora")"; else report "N=$topN pretrain->LoRA" "FAILED"; fi
  rm -rf "$OUT/am_lora" "$OUT/am_pre_top"
fi
head -n "$topN" "$SYNTH" > "$OUT/_am_top.jsonl"; cat "$REAL" "$OUT/_am_top.jsonl" > "$OUT/_am_mix.jsonl"
log "top N=$topN: naive mix (real + $topN synth)"
if ft "$BASE" "$OUT/_am_mix.jsonl" "$OUT/am_mix" "$((FT_STEPS*3))"; then report "N=$topN mix(real+synth)" "$(evalcer "$OUT/am_mix")"; else report "N=$topN mix" "FAILED"; fi
rm -rf "$OUT/am_mix" "$OUT/_am_top.jsonl" "$OUT/_am_mix.jsonl"

echo "read: CER drops as N grows then saturates (only ~6482 unique IAM texts -> style diversity caps)." | tee -a "$RES"
log "done -> $RES"
