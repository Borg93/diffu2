#!/bin/bash
# The first GPU run of the page generator (generate_pages.py): two real Bergskollegium pages refilled with the text of
# two others, every crop drawn with the final v2 LoRA in Bergskollegium's style, read back and labelled. Waits for the
# v2 steering test (run_steer.sh), runs through scripts/spark_guard.sh, then hands the GPU to run_v3.sh. Run detached:
#
#   setsid nohup experiments/flux3_klein/run_pages_smoke.sh >> runs/flux3_pages/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
GT="$HOME/Desktop/data/GT-Lejonet_HTR_202602/GT-RA_bergskollegium_relationer_och_skrivelser_20260220"
VOLUME="$GT/Bergskollegium,_Relationer_och_skrivelser,_E3_5_(1691-1692)/Bergskollegium,_Relationer_och_skrivelser,_E3_5_(1691-1692)/page"
OUT="$REPO/runs/flux3_pages/smoke"
export CUDA_VISIBLE_DEVICES=0
step() { echo "=== $(date '+%F %T') $*"; }

step "waiting for the v2 steering test to finish"
until [ -f "$REPO/runs/flux3_layout_v2/steer/DONE" ]; do
    pgrep -f "flux3_klein/run_(v2|steer)\.s[h]" > /dev/null || { echo "the v2 run or steering test stopped"; exit 1; }
    sleep 60
done

lora="$(ls -d "$REPO"/runs/flux3_layout_lora_v2/epoch_*_step_* | sort -V | tail -1)/model"
step "1/2 generate pages with $lora"
mkdir -p "$OUT"
cd "$AUTOMODEL"
if "$REPO/scripts/spark_guard.sh" --need-gb 45 --name flux-pages-smoke --log "$REPO/runs/flux3_pages/guard.log" -- \
    .venv-managed/bin/python "$REPO/experiments/flux3_klein/generate_pages.py" \
    --templates "$VOLUME/40004026_00128.xml" "$VOLUME/40004026_00094.xml" \
    --text-from "$VOLUME/40004026_00339.xml" "$VOLUME/40004026_00200.xml" \
    --lora "$lora" --out "$OUT" \
    --title "the reports and letters of Bergskollegium" --years "1691–1692" --paper cream --ink "dark brown" \
    > "$OUT/generate.log" 2>&1; then
    tail -3 "$OUT/generate.log"
else
    echo "generation failed, see $OUT/generate.log"
fi
touch "$OUT/DONE"

step "2/2 hand the GPU to run_v3.sh"
cd "$REPO"
exec experiments/flux3_klein/run_v3.sh
