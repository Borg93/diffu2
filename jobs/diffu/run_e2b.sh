#!/bin/bash
# E2b: does more data and training make diffu's letters form? E2's glyph arm (runs/diffu_s_e2_glyph/model) continued on
# all 11,329 v2 training crops at the 512-px stage, 10,000 steps of 8 (packages/train/configs/diffu/diffu_s_e2b_glyph.yaml).
# Cached, smoke-tested (5 steps; the run stops if NeMo's log lacks the caption dropout line), trained, then the step-5,000
# checkpoint and the last are exported (diffu-export) and drawn on E2's 24 held-out windows with their glyph maps
# (data_flux3/e2_val_eval/glyphs), so E2 (step 3,000), 5,000 and 10,000 show the trend. One GPU job at a time through
# jobs/spark_guard.sh. Run detached:
#
#   mkdir -p runs/diffu_e2b && setsid nohup jobs/diffu/run_e2b.sh >> runs/diffu_e2b/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2"
INIT="$REPO/runs/diffu_s_e2_glyph/model"
EVAL="$REPO/data_flux3/e2_val_eval/glyphs"
CACHE="$REPO/data_flux3/cache_diffu_e2b_glyph"
CONFIG="$REPO/packages/train/configs/diffu/diffu_s_e2b_glyph.yaml"
OUT="$REPO/runs/diffu_s_e2b_glyph"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" DIFFU_INIT="$INIT" DIFFU_CACHE="$CACHE"

[ -f "$INIT/modular_model_index.json" ] || { echo "no E2 glyph export at $INIT: run jobs/diffu/run_e2.sh first"; exit 1; }
[ -f "$EVAL/metadata.jsonl" ] || { echo "no held-out glyph windows at $EVAL: run jobs/diffu/run_e2.sh first"; exit 1; }
step() { echo "=== $(date '+%F %T') $*"; }
mkdir -p "$OUT"

step "cache all training crops at scale 0.5 (diffu_train.cache_glyphs)"
if [ ! -f "$CACHE/metadata.json" ]; then  # metadata.json is written last: it marks a finished cache
    "$GUARD" --need-gb 30 --name diffu-e2b-cache --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_glyphs --crops "$CROPS/train" --model "$INIT" --out "$CACHE" --scale 0.5 \
        > "$OUT/cache.log" 2>&1
fi

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    (cd "$AUTOMODEL" && "$GUARD" --need-gb 40 --name "diffu-e2b-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" \
        --trackio.name "diffu_s_e2b_glyph_$name" "$@" > "$OUT/$name.log" 2>&1)
}

step "a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "caption dropout: 0.1 " "$OUT/smoke.log" || { step "the smoke run trained without caption dropout"; exit 1; }

step "train"
train train

draw() {  # draw <checkpoint> <name> <guidance>
    local model="$OUT/model_$2"
    [ -d "$model" ] || CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$1" --init "$INIT" --out "$model"
    if "$GUARD" --need-gb 30 --name "diffu-e2b-eval-$2-g$3" --log "$OUT/guard.log" -- \
        "$REPO/.venv/bin/python" -m diffu_page.cli.evaluate --crops "$EVAL" --model "$model" -n 1000 \
        --steps 50 --guidance "$3" --out "$OUT/eval_$2_g$3" > "$OUT/eval_$2_g$3.log" 2>&1; then
        tail -1 "$OUT/eval_$2_g$3.log"
    else
        step "eval of $2 at guidance $3 FAILED, see $OUT/eval_$2_g$3.log"
    fi
}

step "the held-out windows: step 5,000 and the last checkpoint"
draw "$(ls -d "$OUT"/epoch_*_step_4999)" step5000 4
final=$(ls -d "$OUT"/epoch_*_step_* | sort -V | tail -1)
draw "$final" final 4
draw "$final" final 1

step "done"
touch "$REPO/runs/diffu_e2b_DONE"
