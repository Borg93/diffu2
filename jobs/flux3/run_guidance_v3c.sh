#!/bin/bash
# v3c's final LoRA on the by-example set at a lower guidance scale (2.5, 3.5; run_v3c.sh evaluated it at 4.0).
# Classifier-free guidance amplifies the caption against the empty prompt and leaves the reference out of the
# contrast, so lowering it may let the reference's hand through: each results.json scores CER and how closely the
# drawn hands follow their references. One GPU job at a time through jobs/spark_guard.sh. Run detached:
#
#   setsid nohup jobs/flux3/run_guidance_v3c.sh >> runs/flux3_style_v3c/guidance.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$HOME/Desktop/repos/Automodel/.venv-managed/bin/python"
GUARD="$REPO/jobs/spark_guard.sh"
PAIRS="$REPO/data_flux3/style_pairs_v3c"
OUT="$REPO/runs/flux3_style_v3c"
CKPT="$REPO/runs/flux3_style_ref_lora_v3c"
export CUDA_VISIBLE_DEVICES=0

final=$(ls -d "$CKPT"/epoch_*_step_* | sort -V | tail -1)
for guidance in 2.5 3.5; do
    name="by_example_step_${final##*_}_guidance_$guidance"
    echo "=== $(date '+%F %T') $name"
    if "$GUARD" --need-gb 45 --name "flux-v3c-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$PAIRS/by_example" -n 1000 --steps 50 --guidance "$guidance" \
        --out "$OUT/eval_$name" --lora "v3=$final/model" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
done
