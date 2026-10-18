#!/usr/bin/env bash
# The decisive settle for "does diffu synth help a DATA-RICH IAM recognizer?" — kills the two confounds the
# adversarial audit found (single-seed noise + epochs-not-steps) and pulls the one un-pulled lever (synth
# photometric harmonization). Three arms, STEP-matched (--max-steps, so same optimizer budget + same LR decay)
# and run over N seeds (--seed) for error bars:
#   A) real-only            2000 real
#   B) raw 0.25:1           2000 real + 500 RAW synth        (off-manifold: dense/high-contrast)
#   C) harmonized 0.25:1    2000 real + 500 HARMONIZED synth (same 500 texts, photometrics matched to real)
# Read: if C < A but B ~ A  => the ceiling is a FIXABLE image covariate shift. If neither beats A => the
# data-rich ceiling is real (redundant in-domain text on an already-expert base).
#
#   GPU=2 scripts/exp_htr_harmonize_sweep.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; SEEDS=${SEEDS:-"1 2 3 4 5"}; MAXSTEPS=${MAXSTEPS:-1250}; NSYN=${NSYN:-500}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$OUT/synth_harmonized/train.jsonl" ] || { echo "run scripts/harmonize_synth.py first"; exit 1; }

# build the three train manifests once (B and C share the SAME 500 texts — only image photometrics differ)
head -n "$NSYN" "$OUT/synth_fixed/train.jsonl"      > "$OUT/_syn${NSYN}_raw.jsonl"
head -n "$NSYN" "$OUT/synth_harmonized/train.jsonl" > "$OUT/_syn${NSYN}_harm.jsonl"
cp "$OUT/real/train.jsonl"                             "$OUT/_arm_real.jsonl"
cat "$OUT/real/train.jsonl" "$OUT/_syn${NSYN}_raw.jsonl"  > "$OUT/_arm_raw.jsonl"
cat "$OUT/real/train.jsonl" "$OUT/_syn${NSYN}_harm.jsonl" > "$OUT/_arm_harm.jsonl"

RES="$OUT/results_harmonize.txt"; : > "$RES"
for s in $SEEDS; do
  for arm in real raw harm; do
    od="$OUT/hsweep_${arm}_s${s}"
    log "seed=$s arm=$arm  (max_steps=$MAXSTEPS)"
    uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/_arm_${arm}.jsonl" \
      --out "$od" --max-steps "$MAXSTEPS" --seed "$s" >/dev/null 2>&1
    cer=$(uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$od" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}')
    echo "seed=$s arm=$arm cer=$cer" | tee -a "$RES"
    rm -rf "$od"   # don't keep 15 checkpoints
  done
done

log "aggregating -> $RES"
uv run python - "$RES" <<'PY' | tee -a "$RES"
import sys, collections, statistics as st
rows=[dict(kv.split("=") for kv in l.split()) for l in open(sys.argv[1]) if l.startswith("seed=")]
by=collections.defaultdict(list)
for r in rows: by[r["arm"]].append(float(r["cer"]))
print("\n=== HARMONIZE SWEEP (step-matched, multi-seed) — CER on 1000 real IAM test ===")
base=st.mean(by["real"]) if by.get("real") else None
for arm in ["real","raw","harm"]:
    v=by.get(arm,[])
    if not v: continue
    m=st.mean(v); sd=st.pstdev(v) if len(v)>1 else 0.0
    delta=f"  Δ vs real {100*(m-base):+.2f} pt" if base is not None and arm!="real" else ""
    print(f"{arm:5s}  n={len(v)}  CER {100*m:.2f}% ± {100*sd:.2f}{delta}   {[round(100*x,2) for x in v]}")
print("\nread: harm<real & raw~real => data-rich ceiling is a FIXABLE covariate shift; "
      "neither<real => ceiling is real (saturated expert base).")
PY
log "done -> $RES"
