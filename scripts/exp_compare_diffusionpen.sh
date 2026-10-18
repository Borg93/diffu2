#!/usr/bin/env bash
# diffu vs DiffusionPen as HTR TRAINING DATA (Protocol B — the metric that matters). Both models render the
# SAME IAM line texts; we train an HTR recognizer on real + each synth (equal budget) and compare CER +
# the error buckets. DiffusionPen is word-level, so its lines are its OWN word->line composition (fair: its
# choice, and lines are what a line-recognizer needs). DP_CMD is the DiffusionPen generator hook (set from
# the isolated-env gen_manifest.py once it's verified).
#
#   GPU=2 DP_CMD='cd /home/blackwell/Desktop/DiffusionPen && ./.venv/bin/python gen_manifest.py --device cuda:2' scripts/exp_compare_diffusionpen.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; CMP="$OUT/compare"
BASE=${BASE:-microsoft/trocr-base-stage1}
REAL="$OUT/real_all/train.jsonl"; TEST="$OUT/test/test.jsonl"
DIFFU=checkpoints/exp_sd35_128/step_120000/model.safetensors; ADAPTER=${ADAPTER:-runs/iam_lora_scaleup/final}
GPU=${GPU:-2}; SEED=${SEED:-1}; N=${N:-20000}; FT_STEPS=${FT_STEPS:-6000}; MIX_STEPS=${MIX_STEPS:-9000}
DP_CMD=${DP_CMD:?set DP_CMD to the DiffusionPen gen_manifest.py invocation (isolated env)}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%T)] $*"; }
ft(){ uv run diffu-htr-train --base "$1" --train "$2" --out "$3" --max-steps "$4" --seed "$SEED" >/dev/null 2>&1; }
ecls(){ CUDA_VISIBLE_DEVICES=$GPU uv run python scripts/error_analysis.py --htr "$1" --manifest "$TEST" --out "$2" >/dev/null 2>&1; }

# 0) matched texts
[ -f "$CMP/lines.txt" ] || uv run python scripts/sample_iam_texts.py --n-lines "$N" --out "$CMP"

# 1) render the SAME lines with each model
log "render $N lines with DIFFU"
[ -f "$CMP/diffu_lines/train.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU uv run diffu-synth --ckpt "$DIFFU" --lora "$ADAPTER" \
  --hf-dataset Teklia/IAM-line --texts "$CMP/lines.txt" --out "$CMP/diffu_lines" >/dev/null 2>&1
log "render $N lines with DIFFUSIONPEN"
[ -f "$CMP/dp_lines/train.jsonl" ] || ( eval "$DP_CMD --texts $(pwd)/$CMP/lines.txt --mode line --out $(pwd)/$CMP/dp_lines" ) >/dev/null 2>&1

# 2) three arms: real-only baseline, real+diffu, real+dp  (identical except the synth source)
cat "$REAL" "$CMP/diffu_lines/train.jsonl" > "$OUT/_cmp_diffu.jsonl"
cat "$REAL" "$CMP/dp_lines/train.jsonl"    > "$OUT/_cmp_dp.jsonl"
log "arm 0: real-only ($FT_STEPS)";           ft "$BASE" "$REAL"             "$OUT/cmp_real"  "$FT_STEPS"; ecls "$OUT/cmp_real"  "$OUT/er_cmp_real.json";  rm -rf "$OUT/cmp_real"
log "arm D: real + diffu synth ($MIX_STEPS)";  ft "$BASE" "$OUT/_cmp_diffu.jsonl" "$OUT/cmp_D" "$MIX_STEPS"; ecls "$OUT/cmp_D" "$OUT/er_cmp_D.json"; rm -rf "$OUT/cmp_D"
log "arm P: real + DiffusionPen synth ($MIX_STEPS)"; ft "$BASE" "$OUT/_cmp_dp.jsonl" "$OUT/cmp_P" "$MIX_STEPS"; ecls "$OUT/cmp_P" "$OUT/er_cmp_P.json"; rm -rf "$OUT/cmp_P"

# 3) tabulate real-only vs +diffu vs +DiffusionPen
RES="$OUT/results_compare_dp.txt"
uv run python - "$OUT/er_cmp_real.json" "$OUT/er_cmp_D.json" "$OUT/er_cmp_P.json" <<'PY' | tee "$RES"
import json, sys
R, D, P = (json.load(open(a)) for a in sys.argv[1:4])
print("=== HTR training data: real-only vs +diffu vs +DiffusionPen (same budget) — CER on 1000 IAM test ===")
print(f"{'metric':16} {'real-only':>10} {'+diffu':>9} {'+DiffPen':>9}")
def row(name, r, d, p): print(f"{name:16} {100*r:10.2f} {100*d:9.2f} {100*p:9.2f}")
row("overall CER", R["overall_cer"], D["overall_cer"], P["overall_cer"])
for k in ("digit","upper","punct","lower"):
    row(f"{k} char-err", R["class_error_rate"][k], D["class_error_rate"][k], P["class_error_rate"][k])
print("\nread: lower is better; whichever synth cuts CER more is the better training data.")
PY
rm -f "$OUT/_cmp_diffu.jsonl" "$OUT/_cmp_dp.jsonl"
log "done -> $RES"
