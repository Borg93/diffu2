#!/bin/bash
# The diffu-S smoke run (design study E1): our own model from random weights, conditioned on text only. One GPU job at
# a time, each through jobs/spark_guard.sh:
#   1. diffu-init: diffu-S with adaLN-Zero weights (CPU)                          runs/diffu_s_smoke/init
#   2. cache the first 2,000 v2 training crops with diffu's own encoders           data_flux3/cache_diffu_smoke
#   3. a 5-step training run: fails fast, and its log shows memory and step time
#   4. train 2,000 steps x 8 crops (packages/train/configs/diffu/diffu_s_smoke.yaml)  runs/diffu_s_smoke
#   5. the last checkpoint as a diffu repo (diffu-export), on the 24 held-out crops  runs/diffu_s_smoke/eval_final
# A failed eval is logged; failed caching or training stops the run. A rerun skips a finished init or cache. Needs
# jobs/setup_nemo_env.sh to have installed diffu, diffu-page and diffu-train into NeMo's environment. Run detached:
#
#   mkdir -p runs/diffu_s_smoke && setsid nohup jobs/diffu/run_smoke.sh >> runs/diffu_s_smoke/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2"
OUT="$REPO/runs/diffu_s_smoke"
INIT="$OUT/init"
CACHE="$REPO/data_flux3/cache_diffu_smoke"
CONFIG="$REPO/packages/train/configs/diffu/diffu_s_smoke.yaml"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" DIFFU_INIT="$INIT" DIFFU_CACHE="$CACHE"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 40 --name "diffu-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" --trackio.name "diffu_s_$name" "$@" \
        > "$OUT/$name.log" 2>&1
}

step "1/5 diffu-init"
if [ -f "$INIT/modular_model_index.json" ]; then
    echo "already initialised: $INIT"
else
    CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-init" --out "$INIT" --size S
fi

step "2/5 cache 2,000 crops with diffu's encoders"
# cache_diffu writes metadata.json last, so it marks a finished cache; a few unreadable crops don't stop the run
if [ -f "$CACHE/metadata.json" ]; then
    echo "already cached: $CACHE"
else
    "$GUARD" --need-gb 30 --name diffu-cache --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_diffu --crops "$CROPS/train" --model "$INIT" --out "$CACHE" --limit 2000 \
        > "$OUT/cache.log" 2>&1 || true
    [ -f "$CACHE/metadata.json" ] || { echo "caching failed, see $OUT/cache.log"; exit 1; }
    tail -2 "$OUT/cache.log"
fi

step "3/5 a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=" "$OUT/smoke.log" | tail -1
tail -1 "$OUT/guard.log"

step "4/5 train diffu-S"
train train

step "5/5 the final checkpoint on the held-out crops"
# the init repo's components, the trained transformer and a scheduler at the shift the run trained at
final=$(ls -d "$OUT"/epoch_*_step_* | sort -V | tail -1)
rm -rf "$OUT/model"  # an earlier export: links and configs only
CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$final" --init "$INIT" --out "$OUT/model"
if "$GUARD" --need-gb 30 --name diffu-eval --log "$OUT/guard.log" -- \
    "$PY" -m diffu_page.cli.evaluate --crops "$CROPS/val_eval" --model "$OUT/model" -n 1000 --steps 50 \
    --out "$OUT/eval_final" > "$OUT/eval_final.log" 2>&1; then
    tail -1 "$OUT/eval_final.log"
else
    echo "eval failed, see $OUT/eval_final.log"
fi

step "done"
touch "$OUT/DONE"
