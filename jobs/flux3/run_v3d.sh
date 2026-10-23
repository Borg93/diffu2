#!/bin/bash
# v3d: v3c's same-hand reference pairs, each caption describing the reference's measured hand, ink and paper in
# words (diffu_page.data.style_pairs --caption described), from the crops rebuilt with today's archive names
# (data_flux3/layout_crops_v2b); caption dropout 0.1 on the real empty prompt (diffu_train.dropout); a constant
# learning rate; 3,000 steps; val_loss every 100 steps. Why: FLUX.3's guidance follows the caption, and v3 and v3b drew
# black, bold ink whatever the reference's (diffu_page.cli.hands on their held-out runs). Everything is in
# packages/train/configs/flux3/flux3_style_ref_lora_v3d.yaml. One GPU job at a time through jobs/spark_guard.sh:
#   1. build the pairs (CPU) if they are not built                      data_flux3/style_pairs_v3d
#   2. cache the pairs, the validation pairs and the empty prompt        data_flux3/cache_style_v3d(_val)
#   3. a 5-step smoke run that also scores val_loss once
#   4. train the LoRA                                                   runs/flux3_style_ref_lora_v3d
#   5. the final and middle checkpoints on the by-example set, the final on the held-out crops, each scored for CER
#      and for how closely the drawn hands follow their references (results.json)  runs/flux3_style_v3d/eval_*
# Run detached:
#
#   mkdir -p runs/flux3_style_v3d && setsid nohup jobs/flux3/run_v3d.sh >> runs/flux3_style_v3d/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/jobs/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2b"
PAIRS="$REPO/data_flux3/style_pairs_v3d"
CACHE="$REPO/data_flux3/cache_style_v3d"
VAL_CACHE="$REPO/data_flux3/cache_style_v3d_val"
CONFIG="$REPO/packages/train/configs/flux3/flux3_style_ref_lora_v3d.yaml"
OUT="$REPO/runs/flux3_style_v3d"
CKPT="$REPO/runs/flux3_style_ref_lora_v3d"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 60 --name "flux-v3d-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" --trackio.name "flux3_v3d_$name" \
        "$@" > "$OUT/$name.log" 2>&1
}

evaluate() {  # evaluate <name> <crops dir> <lora dir>
    local name=$1 crops=$2 lora=$3
    if "$GUARD" --need-gb 45 --name "flux-v3d-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$crops" -n 1000 --steps 50 \
        --out "$OUT/eval_$name" --lora "v3d=$lora" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

cache() {  # cache <manifest> <cache dir> <name>: the encoder writes the merged metadata.json last; the empty prompt after it
    local manifest=$1 dir=$2 name=$3
    if [ -f "$dir/metadata.json" ] && [ -f "$dir/empty_prompt.pt" ]; then
        echo "already cached: $dir"
        return
    fi
    rm -rf "$dir"
    "$GUARD" --need-gb 30 --name "flux-v3d-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_pairs --manifest "$manifest" --out "$dir" --model "$MODEL" > "$OUT/$name.log" 2>&1
}

step "1/5 build the pairs"
if [ -f "$PAIRS/stale.json" ]; then  # written last
    echo "already built: $PAIRS"
else
    "$REPO/.venv/bin/python" -m diffu_page.data.style_pairs --reference page --caption described --val-pairs 64 \
        --crops "$CROPS" --out "$PAIRS" > "$OUT/pairs.log" 2>&1
fi
tail -2 "$OUT/pairs.log" 2>/dev/null || true

step "2/5 cache the pairs, the validation pairs and the empty prompt"
cache "$PAIRS/train.jsonl" "$CACHE" cache
cache "$PAIRS/val/val.jsonl" "$VAL_CACHE" cache_val

step "3/5 a 5-step smoke run that scores val_loss once"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --step_scheduler.val_every_steps 5 \
    --lr_scheduler.lr_warmup_steps 1 --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=\|val_loss" "$OUT/smoke.log" | tail -2
tail -1 "$OUT/guard.log"

step "4/5 train the LoRA"
train lora --checkpoint.checkpoint_dir "$CKPT"

step "5/5 evaluate the final and middle checkpoints"
mapfile -t ckpts < <(ls -d "$CKPT"/epoch_*_step_* | sort -V)
middle=${ckpts[$(( ${#ckpts[@]} / 2 ))]}
final=${ckpts[-1]}
evaluate "by_example_step_${final##*_}" "$PAIRS/by_example" "$final/model"
"$REPO/.venv/bin/python" -m diffu_page.cli.sheets \
    --steer "$PAIRS/by_example" --run "$OUT/eval_by_example_step_${final##*_}" || echo "contact sheets failed"
evaluate "val_step_${final##*_}" "$PAIRS/val_eval" "$final/model"
evaluate "by_example_step_${middle##*_}" "$PAIRS/by_example" "$middle/model"

step "done"
touch "$OUT/DONE"
