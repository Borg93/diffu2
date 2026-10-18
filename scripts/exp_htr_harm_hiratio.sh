#!/usr/bin/env bash
# The last open thread: at 0.25:1 the settle found harmonized ~= raw (the photometric covariate shift didn't
# bind at a low ratio). But at a HIGH ratio the synth dominates the batch, so if the covariate shift is what
# makes "too much synth hurt," harmonizing should help MOST here. Test at 2:1 (real=2000 + 4000 synth),
# step-matched (--max-steps 1250) x 5 seeds, raw vs harmonized. Paired within-seed contrast harm-raw is the
# signal. (Real-only baseline for the 5 seeds is already 9.95%+/-0.24 from results_harmonize.txt.)
#
#   GPU=2 scripts/exp_htr_harm_hiratio.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; SEEDS=${SEEDS:-"1 2 3 4 5"}; MAXSTEPS=${MAXSTEPS:-1250}; NSYN=${NSYN:-4000}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$OUT/synth_harmonized/train.jsonl" ] || { echo "run scripts/harmonize_synth.py first"; exit 1; }

# 2:1 arms (raw vs harmonized share the SAME NSYN texts — only photometrics differ)
head -n "$NSYN" "$OUT/synth_fixed/train.jsonl"      > "$OUT/_hi_syn_raw.jsonl"
head -n "$NSYN" "$OUT/synth_harmonized/train.jsonl" > "$OUT/_hi_syn_harm.jsonl"
cat "$OUT/real/train.jsonl" "$OUT/_hi_syn_raw.jsonl"  > "$OUT/_hi_arm_raw.jsonl"
cat "$OUT/real/train.jsonl" "$OUT/_hi_syn_harm.jsonl" > "$OUT/_hi_arm_harm.jsonl"

RES="$OUT/results_harm_hiratio.txt"; : > "$RES"
for s in $SEEDS; do
  for arm in raw harm; do
    od="$OUT/hi_${arm}_s${s}"
    log "2:1  seed=$s arm=$arm  (max_steps=$MAXSTEPS)"
    uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/_hi_arm_${arm}.jsonl" \
      --out "$od" --max-steps "$MAXSTEPS" --seed "$s" >/dev/null 2>&1
    cer=$(uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$od" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}')
    echo "seed=$s arm=$arm cer=$cer" | tee -a "$RES"
    rm -rf "$od"
  done
done

log "aggregating -> $RES"
uv run python - "$RES" <<'PY' | tee -a "$RES"
import sys, collections, statistics as st
rows=[dict(kv.split("=") for kv in l.split()) for l in open(sys.argv[1]) if l.startswith("seed=")]
by=collections.defaultdict(dict)
for r in rows: by[r["seed"]][r["arm"]]=float(r["cer"])
raw=[by[s]["raw"] for s in by if "raw" in by[s]]
harm=[by[s]["harm"] for s in by if "harm" in by[s]]
paired=[by[s]["harm"]-by[s]["raw"] for s in by if "harm" in by[s] and "raw" in by[s]]
print("\n=== 2:1 HARMONIZE (step-matched, multi-seed) — CER on 1000 real IAM test ===")
print(f"  reference: real-only 9.95% +/- 0.24 (from results_harmonize.txt)")
for name,v in [("raw 2:1 ",raw),("harm 2:1",harm)]:
    print(f"{name}  n={len(v)}  CER {100*st.mean(v):.2f}% +/- {100*st.pstdev(v):.2f}   {[round(100*x,2) for x in v]}")
if paired:
    m=st.mean(paired); sd=st.pstdev(paired)
    better=sum(1 for d in paired if d<0)
    print(f"\npaired harm-raw:  {100*m:+.2f} pt +/- {100*sd:.2f}   harm better in {better}/{len(paired)} seeds")
    print("read: harm-raw clearly NEGATIVE => harmonization recovers the high-ratio covariate-shift damage "
          "(the shift binds at high ratio). ~0 => even at 2:1 photometrics are not the limiter.")
PY
log "done -> $RES"
