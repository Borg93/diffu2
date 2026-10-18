#!/usr/bin/env bash
# STAGE 3 (the payoff): can 100k diffu fakes build a good IAM recognizer FROM A BASE THAT NEVER SAW IAM?
# Base = microsoft/trocr-base-stage1 (generic TrOCR pretrain, NOT IAM-finetuned) — so synth-pretraining can
# actually help (on the IAM-expert trocr-base-handwritten it only forgets; proven earlier). Real = all 6482
# IAM. All arms eval on the same held-out 1000-line test. Each arm's CER is written the moment it finishes,
# so an unattended run keeps partial results even if a later arm dies (no `set -e` abort).
#
#   GPU=2 scripts/exp_htr_scaleup_sweep.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab
BASE=${BASE:-microsoft/trocr-base-stage1}          # a NON-IAM base — the whole point
REAL="$OUT/real_all/train.jsonl"                    # all 6482 IAM
SYNTH="$OUT/synth100k/train.jsonl"                  # 100k fakes (Stage 2)
TEST="$OUT/test/test.jsonl"
GPU=${GPU:-2}; SEED=${SEED:-1}
PRE_STEPS=${PRE_STEPS:-20000}    # pretrain on 100k synth (~1.6 epochs)
FT_STEPS=${FT_STEPS:-6000}       # finetune on 6482 real (~7.4 epochs)
MIX_STEPS=${MIX_STEPS:-26000}    # mix arm: match B's total compute (PRE+FT)
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%T)] $*"; }

for f in "$REAL" "$SYNTH" "$TEST"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
mix="$OUT/_scaleup_mix.jsonl"; cat "$REAL" "$SYNTH" > "$mix"

train(){ uv run diffu-htr-train --base "$1" --train "$2" --out "$3" --max-steps "$4" --seed "$SEED" "${@:5}" >/dev/null 2>&1; }
evalcer(){ uv run diffu-eval --manifest "$TEST" --htr "$1" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}'; }
report(){ printf "%-38s %s\n" "$1" "$2" | tee -a "$RES"; }

RES="$OUT/results_scaleup.txt"
{ echo "=== SCALE-UP (base=$BASE, all $(wc -l <"$REAL") IAM real, $(wc -l <"$SYNTH") fakes) — CER on 1000 real IAM test ==="
  echo "config: PRE=$PRE_STEPS FT=$FT_STEPS MIX=$MIX_STEPS seed=$SEED"; } > "$RES"

# A: real-only from the non-IAM base
log "arm A: base -> finetune real ($FT_STEPS)"
if train "$BASE" "$REAL" "$OUT/su_A" "$FT_STEPS"; then report "A real-only" "$(evalcer "$OUT/su_A")"; else report "A real-only" "FAILED"; fi
rm -rf "$OUT/su_A"

# shared synth-pretrain (used by B and C)
log "pretrain: base -> 100k synth ($PRE_STEPS)"
if train "$BASE" "$SYNTH" "$OUT/su_pre" "$PRE_STEPS"; then
  report "(ref) synth-pretrain only" "$(evalcer "$OUT/su_pre")"
  log "arm B: synth-pretrain -> finetune real ($FT_STEPS)"
  if train "$OUT/su_pre" "$REAL" "$OUT/su_B" "$FT_STEPS"; then report "B synth-pretrain->finetune" "$(evalcer "$OUT/su_B")"; else report "B synth-pretrain->finetune" "FAILED"; fi
  rm -rf "$OUT/su_B"
  log "arm C: synth-pretrain -> LoRA real ($FT_STEPS)"
  if train "$OUT/su_pre" "$REAL" "$OUT/su_C" "$FT_STEPS" --lora --lora-rank 16; then report "C synth-pretrain->LoRA" "$(evalcer "$OUT/su_C")"; else report "C synth-pretrain->LoRA" "FAILED"; fi
  rm -rf "$OUT/su_C" "$OUT/su_pre"
else
  report "B/C (pretrain FAILED)" "SKIPPED"; rm -rf "$OUT/su_pre"
fi

# D: naive mix (real + 100k synth), compute-matched to B
log "arm D: mix real+100k synth ($MIX_STEPS)"
if train "$BASE" "$mix" "$OUT/su_D" "$MIX_STEPS"; then report "D mix(real+100k)" "$(evalcer "$OUT/su_D")"; else report "D mix(real+100k)" "FAILED"; fi
rm -rf "$OUT/su_D" "$mix"

echo "read: B/C < A => 100k fake-pretraining builds a better reader from a non-IAM base (the win)." | tee -a "$RES"
log "done -> $RES"
