#!/usr/bin/env bash
# "Do more fakes and test again": the step-matched settle showed more synth is better up to 2:1 (8.45%).
# Push further with a fresh disjoint 12000-line batch (synth_more, seed=1) pooled with the original 4000
# (synth_fixed, seed=0) = 16000, and sweep the ratio UP: 2:1, 4:1, 8:1. Step-matched (--max-steps 1250) x
# 5 seeds, raw synth (harmonization already shown moot). Question: does "more is better" keep holding, or
# does real data get diluted enough to turn the curve over?
#
#   GPU=2 scripts/exp_htr_moreratio.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}
GPU=${GPU:-2}; SEEDS=${SEEDS:-"1 2 3 4 5"}; MAXSTEPS=${MAXSTEPS:-1250}
NSYNS=${NSYNS:-"4000 8000 16000"}   # real=2000 -> ratios 2:1, 4:1, 8:1
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$OUT/synth_more/train.jsonl" ] || { echo "generate synth_more first (diffu-synth --seed 1 --n 12000)"; exit 1; }

# pool = original 4000 (seed 0) + new 12000 (seed 1) = 16000
cat "$OUT/synth_fixed/train.jsonl" "$OUT/synth_more/train.jsonl" > "$OUT/_pool_synth.jsonl"
log "pooled synth: $(wc -l < "$OUT/_pool_synth.jsonl") lines"

RES="$OUT/results_moreratio.txt"; : > "$RES"
for s in $SEEDS; do
  for nsyn in $NSYNS; do
    head -n "$nsyn" "$OUT/_pool_synth.jsonl" > "$OUT/_pool_${nsyn}.jsonl"
    cat "$OUT/real/train.jsonl" "$OUT/_pool_${nsyn}.jsonl" > "$OUT/_more_arm.jsonl"
    od="$OUT/more_${nsyn}_s${s}"
    log "seed=$s nsyn=$nsyn (ratio $((nsyn/2000)):1)  max_steps=$MAXSTEPS"
    uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/_more_arm.jsonl" \
      --out "$od" --max-steps "$MAXSTEPS" --seed "$s" >/dev/null 2>&1
    cer=$(uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$od" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}')
    echo "seed=$s nsyn=$nsyn cer=$cer" | tee -a "$RES"
    rm -rf "$od"
  done
done

log "aggregating -> $RES"
uv run python - "$RES" "$OUT/results_harmonize.txt" <<'PY' | tee -a "$RES"
import sys, collections, statistics as st
rows=[dict(kv.split("=") for kv in l.split()) for l in open(sys.argv[1]) if l.startswith("seed=")]
# paired real-only baseline per seed from the earlier settle
base={}
for l in open(sys.argv[2]):
    if l.startswith("seed=") and "arm=real" in l:
        d=dict(kv.split("=") for kv in l.split()); base[d["seed"]]=float(d["cer"])
by=collections.defaultdict(dict)
for r in rows: by[r["nsyn"]].setdefault(r["seed"],float(r["cer"]))
print("\n=== MORE-FAKES RATIO SWEEP (step-matched 1250, 5 seeds) — CER on 1000 real IAM test ===")
print(f"  real-only baseline (paired, per seed): mean {100*st.mean(list(base.values())):.2f}%")
for nsyn in sorted(by, key=int):
    v=list(by[nsyn].values()); m=st.mean(v); sd=st.pstdev(v)
    pair=[by[nsyn][s]-base[s] for s in by[nsyn] if s in base]
    dm=st.mean(pair); better=sum(1 for x in pair if x<0)
    print(f"ratio {int(nsyn)//2000}:1 ({nsyn:>5} synth)  CER {100*m:.2f}% +/- {100*sd:.2f}   "
          f"paired Δ {100*dm:+.2f} pt  better {better}/{len(pair)}   {[round(100*x,2) for x in v]}")
print("\nread: Δ keeps getting MORE negative => more fakes still win; Δ turns back up => real data too diluted (the peak).")
PY
log "done -> $RES"
