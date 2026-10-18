#!/usr/bin/env bash
# PAYOFF: does error-DRIVEN synth beat BLIND synth at the same budget? Train a baseline, analyze where it
# fails, generate TARGETED synth for those failures, and compare against the SAME number of UNIFORM synth
# lines — measuring overall CER and, crucially, the weak buckets (digit/upper/punct char-error).
#
#   GPU=2 scripts/exp_targeted_vs_uniform.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab
BASE=${BASE:-microsoft/trocr-base-stage1}
REAL="$OUT/real_all/train.jsonl"
POOL="$OUT/synth100k/train.jsonl"            # 200k uniform pool (for the uniform arm)
VAL="$OUT/val/val.jsonl"                      # held-out val for the error analysis (NO test leakage)
TEST="$OUT/test/test.jsonl"                   # only ever used for the FINAL eval
DIFFU=checkpoints/exp_sd35_128/step_120000/model.safetensors
ADAPTER=${ADAPTER:-runs/iam_lora_scaleup/final}
GPU=${GPU:-2}; SEED=${SEED:-1}
BUDGET=${BUDGET:-20000}                       # synth lines each arm gets (uniform vs targeted)
FT_STEPS=${FT_STEPS:-6000}; MIX_STEPS=${MIX_STEPS:-9000}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%T)] $*"; }
for f in "$REAL" "$POOL" "$TEST" "$DIFFU"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done

ft(){ uv run diffu-htr-train --base "$1" --train "$2" --out "$3" --max-steps "$4" --seed "$SEED" >/dev/null 2>&1; }

# 1) baseline "target" model — analyze its errors on VALIDATION (never test) to decide what to synthesize
[ -f "$VAL" ] || { echo "missing $VAL — run: uv run python scripts/prepare_iam_full.py --split validation --out $OUT/val --name val.jsonl"; exit 1; }
log "baseline: $BASE -> finetune real ($FT_STEPS)"
ft "$BASE" "$REAL" "$OUT/tv_base" "$FT_STEPS"
log "error analysis of baseline on VAL (held-out, no test leakage) -> error_report_tv.json"
CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/error_analysis.py --htr "$OUT/tv_base" --manifest "$VAL" --out "$OUT/error_report_tv.json" >/dev/null 2>&1
rm -rf "$OUT/tv_base"

# 2) targeted text list from those errors, then RENDER it with diffu (the expensive GPU step)
log "targeted text resampling (n=$BUDGET)"
uv run python scripts/targeted_synth.py --report "$OUT/error_report_tv.json" --n "$BUDGET" --out "$OUT/targeted_texts.txt" 2>/dev/null
log "generating $BUDGET TARGETED synth lines with diffu (this is the long step)"
CUDA_VISIBLE_DEVICES=$GPU uv run diffu-synth --ckpt "$DIFFU" --lora "$ADAPTER" \
  --hf-dataset Teklia/IAM-line --texts "$OUT/targeted_texts.txt" --out "$OUT/synth_targeted" >/dev/null 2>&1
TGT="$OUT/synth_targeted/train.jsonl"

# 3) equal-budget UNIFORM draw from the 200k pool
head -n "$BUDGET" "$POOL" > "$OUT/_uni.jsonl"
cat "$REAL" "$OUT/_uni.jsonl" > "$OUT/_mix_uni.jsonl"
cat "$REAL" "$TGT"          > "$OUT/_mix_tgt.jsonl"

# 4) two arms, identical except uniform vs targeted synth content
log "arm U: real + $BUDGET UNIFORM synth ($MIX_STEPS)"; ft "$BASE" "$OUT/_mix_uni.jsonl" "$OUT/tv_U" "$MIX_STEPS"
log "arm T: real + $BUDGET TARGETED synth ($MIX_STEPS)"; ft "$BASE" "$OUT/_mix_tgt.jsonl" "$OUT/tv_T" "$MIX_STEPS"

# 5) analyze both -> overall + weak-bucket CER, side by side
CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/error_analysis.py --htr "$OUT/tv_U" --manifest "$TEST" --out "$OUT/error_report_U.json" >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/error_analysis.py --htr "$OUT/tv_T" --manifest "$TEST" --out "$OUT/error_report_T.json" >/dev/null 2>&1

RES="$OUT/results_targeted.txt"
uv run python - "$OUT/error_report_U.json" "$OUT/error_report_T.json" <<'PY' | tee "$RES"
import json, sys
U = json.load(open(sys.argv[1])); T = json.load(open(sys.argv[2]))
print("=== TARGETED vs UNIFORM synth (same budget) — CER on 1000 real IAM test ===")
print(f"{'metric':22} {'uniform':>9} {'targeted':>9} {'delta':>8}")
def row(name, u, t): print(f"{name:22} {100*u:9.2f} {100*t:9.2f} {100*(t-u):+8.2f}")
row("overall CER", U["overall_cer"], T["overall_cer"])
for k in ("digit","upper","punct","lower"):
    row(f"{k} char-err", U["class_error_rate"][k], T["class_error_rate"][k])
print("\nread: targeted should cut the DIGIT/UPPER/PUNCT rows more than uniform at equal synth budget.")
PY
rm -rf "$OUT/tv_U" "$OUT/tv_T" "$OUT/_uni.jsonl" "$OUT/_mix_uni.jsonl" "$OUT/_mix_tgt.jsonl" "$OUT/synth_targeted"
log "done -> $RES"
