#!/bin/bash
# E1b: diffu-S continued from E1's last weights with caption dropout (packages/train/configs/diffu/diffu_s_e1b.yaml):
# does classifier-free guidance work once the model has learned the empty prompt, and do letters start to form? One
# GPU job at a time, each through jobs/spark_guard.sh:
#   1. the empty prompt into E1's cache, made before caches kept it               data_flux3/cache_diffu_smoke
#   2. a 5-step smoke run
#   3. train 3,000 steps x 8 crops                                                 runs/diffu_s_e1b
#   4. the last checkpoint as a diffu repo (diffu-export), drawn on the 24 held-out crops with guidance 4 and without
# A failed eval is logged; anything else that fails stops the run. Needs E1's export (runs/diffu_s_smoke/model, written
# by jobs/diffu/run_smoke.sh). Run detached:
#
#   mkdir -p runs/diffu_s_e1b && setsid nohup jobs/diffu/run_e1b.sh >> runs/diffu_s_e1b/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2"
OUT="$REPO/runs/diffu_s_e1b"
INIT="$REPO/runs/diffu_s_smoke/model"
CACHE="$REPO/data_flux3/cache_diffu_smoke"
CONFIG="$REPO/packages/train/configs/diffu/diffu_s_e1b.yaml"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" DIFFU_INIT="$INIT" DIFFU_CACHE="$CACHE"  # the training config's paths

[ -f "$INIT/modular_model_index.json" ] || { echo "no E1 export at $INIT: run jobs/diffu/run_smoke.sh first"; exit 1; }
mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 40 --name "diffu-e1b-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" --trackio.name "diffu_s_e1b_$name" "$@" \
        > "$OUT/$name.log" 2>&1
}

draw() {  # draw <name> [evaluate options ...]: the held-out crops drawn by the exported model
    local name=$1
    shift
    if "$GUARD" --need-gb 30 --name "diffu-e1b-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$CROPS/val_eval" --model "$OUT/model" -n 1000 --steps 50 \
        --out "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1; then
        tail -1 "$OUT/$name.log"
    else
        echo "$name failed, see $OUT/$name.log"
    fi
}

step "1/4 the empty prompt into E1's cache"
if [ -f "$CACHE/empty_prompt.pt" ]; then
    echo "already there: $CACHE/empty_prompt.pt"
else
    "$GUARD" --need-gb 30 --name diffu-e1b-empty --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.empty_prompt --model "$INIT" --cache "$CACHE" > "$OUT/empty_prompt.log" 2>&1
fi

step "2/4 a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=" "$OUT/smoke.log" | tail -1
grep -a "caption dropout: 0.1 " "$OUT/smoke.log" || { echo "the smoke run trained without caption dropout"; exit 1; }

step "3/4 train"
train train

step "4/4 the last checkpoint on the held-out crops"
final=$(ls -d "$OUT"/epoch_*_step_* | sort -V | tail -1)
rm -rf "$OUT/model"  # an earlier export: links and configs only
CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$final" --init "$INIT" --out "$OUT/model"
draw eval_final
draw eval_final_g1 --guidance 1

step "done"
touch "$OUT/DONE"
