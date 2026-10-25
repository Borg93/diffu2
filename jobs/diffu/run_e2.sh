#!/bin/bash
# E2: does the glyph map make diffu's letters form? Two arms at the 512-px stage from E1b's weights, on the first 2,000
# v2 training crops (E1's), 3,000 steps of 8 each (packages/train/configs/diffu/diffu_s_e2_{text,glyph}.yaml):
#   text   the caption alone, the control                          data_flux3/cache_diffu_e2_text    runs/diffu_s_e2_text
#   glyph  the caption and the crop's glyph map as reference image data_flux3/cache_diffu_e2_glyph   runs/diffu_s_e2_glyph
# Each arm is cached, smoke-tested (5 steps; the run stops if NeMo's log lacks the caption dropout line), trained,
# exported (diffu-export) and drawn at guidance 4 and 1 on the 24 held-out windows at the same scale, the glyph arm with
# their glyph maps (diffu_page.data.glyph_set: data_flux3/e2_val_eval). One GPU job at a time through
# jobs/spark_guard.sh. Needs E1b's export (runs/diffu_s_e1b/model). Run detached:
#
#   mkdir -p runs/diffu_e2 && setsid nohup jobs/diffu/run_e2.sh >> runs/diffu_e2/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2"
INIT="$REPO/runs/diffu_s_e1b/model"
EVAL="$REPO/data_flux3/e2_val_eval"
SCALE=0.5
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" DIFFU_INIT="$INIT"

[ -f "$INIT/modular_model_index.json" ] || { echo "no E1b export at $INIT: run jobs/diffu/run_e1b.sh first"; exit 1; }
step() { echo "=== $(date '+%F %T') $*"; }

step "the held-out windows at scale $SCALE, with their glyph maps"
[ -f "$EVAL/glyphs/metadata.jsonl" ] ||
    "$REPO/.venv/bin/python" -m diffu_page.data.glyph_set --crops "$CROPS/val_eval" --out "$EVAL" --scale "$SCALE"

for arm in text glyph; do
    out="$REPO/runs/diffu_s_e2_$arm"
    cache="$REPO/data_flux3/cache_diffu_e2_$arm"
    config="$REPO/packages/train/configs/diffu/diffu_s_e2_$arm.yaml"
    [ "$arm" = text ] && command=diffu_train.cache_diffu || command=diffu_train.cache_glyphs
    mkdir -p "$out"

    step "$arm: cache 2,000 crops at scale $SCALE ($command)"
    if [ ! -f "$cache/metadata.json" ]; then  # metadata.json is written last: it marks a finished cache
        "$GUARD" --need-gb 30 --name "diffu-e2-$arm-cache" --log "$out/guard.log" -- \
            "$PY" -m "$command" --crops "$CROPS/train" --model "$INIT" --out "$cache" --limit 2000 --scale "$SCALE" \
            > "$out/cache.log" 2>&1
    fi

    export DIFFU_CACHE="$cache"
    train() {  # train <name> [config overrides ...]
        local name=$1
        shift
        (cd "$AUTOMODEL" && "$GUARD" --need-gb 40 --name "diffu-e2-$arm-$name" --log "$out/guard.log" -- \
            "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$config" \
            --trackio.name "diffu_s_e2_${arm}_$name" "$@" > "$out/$name.log" 2>&1)
    }

    step "$arm: a 5-step smoke run"
    rm -rf "$out/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
    train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
        --checkpoint.checkpoint_dir "$out/smoke_ckpt"
    grep -a "caption dropout: 0.1 " "$out/smoke.log" || { step "$arm: the smoke run trained without caption dropout"; exit 1; }

    step "$arm: train"
    train train

    step "$arm: the last checkpoint on the held-out windows"
    final=$(ls -d "$out"/epoch_*_step_* | sort -V | tail -1)
    rm -rf "$out/model"  # an earlier export: links and configs only
    CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$final" --init "$INIT" --out "$out/model"
    [ "$arm" = text ] && set_dir="$EVAL/text" || set_dir="$EVAL/glyphs"
    for guidance in 4 1; do
        if "$GUARD" --need-gb 30 --name "diffu-e2-$arm-eval-g$guidance" --log "$out/guard.log" -- \
            "$REPO/.venv/bin/python" -m diffu_page.cli.evaluate --crops "$set_dir" --model "$out/model" -n 1000 \
            --steps 50 --guidance "$guidance" --out "$out/eval_g$guidance" > "$out/eval_g$guidance.log" 2>&1; then
            tail -1 "$out/eval_g$guidance.log"
        else
            step "$arm: eval at guidance $guidance FAILED, see $out/eval_g$guidance.log"
        fi
    done
done

step "done"
touch "$REPO/runs/diffu_e2_DONE"
