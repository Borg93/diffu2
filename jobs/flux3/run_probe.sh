#!/bin/bash
# The control probe for described reference captions (diffu_page.data.probe_set): base FLUX.3 Klein, no LoRA, draws
# v3c's held-out crops from their own references three ways: captions that name no style (plain), captions that
# describe the reference's measured hand, ink and paper (described), and captions that describe the reference whose
# words differ most (contradicting). Whether the drawn hands move with the words tells if v3d's captions can steer at
# all. One GPU job at a time through jobs/spark_guard.sh; each form's results.json scores the drawn hands against the
# references (runs/probe_described/<form>). Run detached:
#
#   mkdir -p runs/probe_described && setsid nohup jobs/flux3/run_probe.sh >> runs/probe_described/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="$HOME/Desktop/repos/Automodel/.venv-managed/bin/python"
GUARD="$REPO/jobs/spark_guard.sh"
PROBE="$REPO/data_flux3/probe_described"
OUT="$REPO/runs/probe_described"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT"
step() { echo "=== $(date '+%F %T') $*"; }

step "the probe set"
if [ -f "$PROBE/looks.json" ]; then  # written last
    echo "already built: $PROBE"
else
    "$REPO/.venv/bin/python" -m diffu_page.data.probe_set --pairs "$REPO/data_flux3/style_pairs_v3c/val_eval" \
        --out "$PROBE"
fi

for form in plain described contradicting; do
    step "draw $form"
    if "$GUARD" --need-gb 45 --name "flux-probe-$form" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$PROBE/$form" -n 1000 --steps 50 --out "$OUT/$form" \
        > "$OUT/$form.log" 2>&1; then
        tail -1 "$OUT/$form.log"
    else
        echo "probe $form failed, see $OUT/$form.log"
    fi
done

step "done"
touch "$OUT/DONE"
