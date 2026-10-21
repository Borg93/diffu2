#!/bin/bash
# The v3 reference-style LoRA: FLUX.3 Klein draws a crop's text and layout in the hand, ink and paper of a reference
# crop from another page of the same volume (page_synth.data.style_pairs -> data_flux3/style_pairs_v3). Waits for the v2
# steering test (run_steer.sh) to finish, then one GPU job at a time, each through scripts/spark_guard.sh:
#   1. cache the pairs with NeMo's FLUX.3 image-edit encoder             data_flux3/cache_style_v3
#   2. a 5-step training smoke run: fails fast, and its log shows memory and step time
#   3. train the LoRA (flux3_style_ref_lora_v3.yaml)                     runs/flux3_style_ref_lora_v3
#   4. the final checkpoint on the by-example set (+ page_synth.sheets) and the held-out crops, the middle
#      checkpoint on the by-example set                                   runs/flux3_style_v3/eval_*
# A failed eval is logged and skipped; failed caching or training stops the run. A rerun skips a finished cache.
# Run detached:
#
#   setsid nohup jobs/flux3/run_v3.sh >> runs/flux3_style_v3/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/scripts/spark_guard.sh"
PAIRS="$REPO/data_flux3/style_pairs_v3"
CACHE="$REPO/data_flux3/cache_style_v3"
CONFIG="$REPO/configs/flux3/flux3_style_ref_lora_v3.yaml"
OUT="$REPO/runs/flux3_style_v3"
CKPT="$REPO/runs/flux3_style_ref_lora_v3"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 60 --name "flux-v3-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" "$@" \
        > "$OUT/$name.log" 2>&1
}

evaluate() {  # evaluate <name> <crops dir> <lora dir>
    local name=$1 crops=$2 lora=$3
    if "$GUARD" --need-gb 45 --name "flux-v3-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m page_synth.evaluate --crops "$crops" -n 1000 --steps 50 \
        --out "$OUT/eval_$name" --lora "$lora" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

step "waiting for the v2 steering test to finish"
until [ -f "$REPO/runs/flux3_layout_v2/steer/DONE" ]; do
    pgrep -f "run_(v2|steer)\.s[h]" > /dev/null || { echo "the v2 run or steering test stopped"; exit 1; }
    sleep 60
done

step "1/4 cache the pairs"
# the encoder writes the merged metadata.json last, so it marks a finished cache
if [ -f "$CACHE/metadata.json" ]; then
    echo "already cached: $CACHE"
else
    rm -rf "$CACHE"
    "$GUARD" --need-gb 30 --name flux-v3-cache --log "$OUT/guard.log" -- \
        "$PY" -m page_synth.data.cache_pairs --manifest "$PAIRS/train.jsonl" \
        --out "$CACHE" --model "$MODEL" > "$OUT/cache.log" 2>&1
fi

step "2/4 a 5-step smoke run"
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=" "$OUT/smoke.log" | tail -1
tail -1 "$OUT/guard.log"

step "3/4 train the LoRA"
train lora

step "4/4 evaluate the final and middle checkpoints"
mapfile -t ckpts < <(ls -d "$CKPT"/epoch_*_step_* | sort -V)
middle=${ckpts[$(( ${#ckpts[@]} / 2 ))]}
final=${ckpts[-1]}
evaluate "by_example_step_${final##*_}" "$PAIRS/by_example" "$final/model"
"$REPO/.venv/bin/python" -m page_synth.sheets \
    --steer "$PAIRS/by_example" --run "$OUT/eval_by_example_step_${final##*_}" || echo "contact sheets failed"
evaluate "val_step_${final##*_}" "$PAIRS/val_eval" "$final/model"
evaluate "by_example_step_${middle##*_}" "$PAIRS/by_example" "$middle/model"

step "done"
touch "$OUT/DONE"
