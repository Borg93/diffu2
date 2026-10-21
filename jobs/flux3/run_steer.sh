#!/bin/bash
# The v2 steering test (page_synth.data.steer_set -> data_flux3/steer_v2): two held-out crops' text and layout drawn as six
# archives and three scan looks with the final v2 LoRA, every variant of a crop from the same noise, then
# page_synth.sheets contact sheets (runs/flux3_layout_v2/steer/steer_template_<t>.jpg). Waits for run_v2.sh to
# finish; the GPU step goes through scripts/spark_guard.sh. Run detached:
#
#   setsid nohup jobs/flux3/run_steer.sh >> runs/flux3_layout_v2/steer_run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
GUARD="$REPO/scripts/spark_guard.sh"
STEER="$REPO/data_flux3/steer_v2"
OUT="$REPO/runs/flux3_layout_v2/steer"
export CUDA_VISIBLE_DEVICES=0

step() { echo "=== $(date '+%F %T') $*"; }

step "waiting for run_v2.sh to finish"
until [ -f "$REPO/runs/flux3_layout_v2/DONE" ]; do
    pgrep -f "run_v2\.s[h]" > /dev/null || { echo "run_v2.sh stopped without finishing"; exit 1; }
    sleep 60
done

lora="$(ls -d "$REPO"/runs/flux3_layout_lora_v2/epoch_*_step_* | sort -V | tail -1)/model"
step "1/2 the steering prompts with $lora"
mkdir -p "$OUT"
cd "$AUTOMODEL"  # the generator runs in the NeMo AutoModel environment
"$GUARD" --need-gb 40 --name flux-v2-steer --log "$REPO/runs/flux3_layout_v2/guard.log" -- \
    "$PY" -m page_synth.evaluate --crops "$STEER" -n 1000 --steps 50 \
    --out "$OUT" --lora "$lora" > "$OUT/eval.log" 2>&1
tail -1 "$OUT/eval.log"

step "2/2 contact sheets"
"$REPO/.venv/bin/python" -m page_synth.sheets --steer "$STEER" --run "$OUT"

step "done"
touch "$OUT/DONE"
