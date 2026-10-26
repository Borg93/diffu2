#!/bin/bash
# E2c: the glyph map by channel concatenation (scratchpad glyph_concat/plan.md, as diffusers' flux-control recipe trains
# FLUX.1 Canny and Depth). E2b gives diffu the glyph map as a reference image, which reaches the image tokens only through
# joint attention; a glyph model takes the map's latents beside its own on channels, each image token with its own cell
# of the map. Writes diffu-S as a glyph model from E2b's own start (runs/diffu_s_vl_init's weights, its image input and
# output widened with zeros for the map: diffu-init --glyph), trains it on E2b's cache (the map is each pair's one
# context latents) with E2b's settings and learning-rate curve for 5,000 steps
# (packages/train/configs/diffu/diffu_s_e2c_glyphcat.yaml) through diffu's own entry point (diffu_train.finetune: NeMo's
# recipe with the glyph adapter), then draws its last checkpoint on the held-out windows (guidance 4 and 1) and the held-in
# crops (guidance 4), each from its caption and the glyph map drawn from its plan (data_flux3/e2_*_eval/text), to set
# beside E2b's and E2n's step 5,000. The argument names the latent stats it starts with, whichever normalization won E2n:
#   pages  our pages' (runs/diffu_s_e2n_glyph/latent_stats.json), as E2n
#   qwen   Qwen-Image's, as E2b
# Every checkpoint is consolidated, so any step can be drawn. One GPU job at a time through jobs/spark_guard.sh. Run
# detached:
#
#   mkdir -p runs/diffu_e2c && setsid nohup jobs/diffu/run_e2c.sh pages >> runs/diffu_e2c/run.log 2>&1 < /dev/null &

set -euo pipefail

ARM="${1:-}"
case "$ARM" in
    pages | qwen) ;;
    *) echo "usage: run_e2c.sh pages|qwen (the latent stats to start with: our pages' or Qwen-Image's)"; exit 2 ;;
esac

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
START="$REPO/runs/diffu_s_vl_init"
CACHE="$REPO/data_flux3/cache_diffu_e2b_glyph"
STATS="$REPO/runs/diffu_s_e2n_glyph/latent_stats.json"
INIT="$REPO/runs/diffu_s_e2c_${ARM}_init"
CONFIG="$REPO/packages/train/configs/diffu/diffu_s_e2c_glyphcat.yaml"
OUT="$REPO/runs/diffu_s_e2c_glyphcat_$ARM"
HELD_OUT="$REPO/data_flux3/e2_val_eval/text"
HELD_IN="$REPO/data_flux3/e2_train_eval/text"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_INIT="$INIT" DIFFU_CACHE="$CACHE" DIFFU_OUT="$OUT"

[ -f "$CACHE/metadata.json" ] || { echo "no finished cache at $CACHE: run jobs/diffu/run_e2b.sh first"; exit 1; }
[ -f "$START/modular_model_index.json" ] || { echo "no E2b start at $START: run jobs/diffu/run_e2b.sh first"; exit 1; }
[ "$ARM" = qwen ] || [ -f "$STATS" ] || { echo "no latent stats at $STATS: run jobs/diffu/run_e2n.sh first"; exit 1; }
for set_dir in "$HELD_OUT" "$HELD_IN"; do
    [ -f "$set_dir/metadata.jsonl" ] || { echo "no eval crops at $set_dir"; exit 1; }
done
step() { echo "=== $(date '+%F %T') $*"; }
mkdir -p "$OUT"

step "diffu-S as a glyph model from E2b's start, with $ARM latent stats"
stats=()
[ "$ARM" = pages ] && stats=(--latent-stats "$STATS")
[ -f "$INIT/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-init" --size S --glyph --out "$INIT" --weights-from "$START" \
        "${stats[@]}"

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    (cd "$AUTOMODEL" && "$GUARD" --need-gb 40 --name "diffu-e2c-$ARM-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 -m diffu_train.finetune -c "$CONFIG" \
        --trackio.name "diffu_s_e2c_glyphcat_${ARM}_$name" "$@" > "$OUT/$name.log" 2>&1)
}

draw() {  # draw <model> <eval set> <name> <guidance> [evaluate options ...]: fails as the evaluation does
    local model=$1 set_dir=$2 name=$3 guidance=$4
    shift 4
    if "$GUARD" --need-gb 30 --name "diffu-e2c-$ARM-eval-$name-g$guidance" --log "$OUT/guard.log" -- \
        "$REPO/.venv/bin/python" -m diffu_page.cli.evaluate --crops "$set_dir" --model "$model" --guidance "$guidance" \
        --out "$OUT/eval_${name}_g$guidance" "$@" > "$OUT/eval_${name}_g$guidance.log" 2>&1; then
        tail -1 "$OUT/eval_${name}_g$guidance.log"
    else
        step "eval of $name at guidance $guidance FAILED, see $OUT/eval_${name}_g$guidance.log"
        return 1
    fi
}

step "a 5-step smoke run, its checkpoint drawn on one held-out window"
rm -rf "$OUT/smoke_ckpt" "$OUT/model_smoke"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "caption dropout: 0.1 " "$OUT/smoke.log" || { step "the smoke run trained without caption dropout"; exit 1; }
grep -a "glyph concat: " "$OUT/smoke.log" || { step "the smoke run trained without the glyph adapter"; exit 1; }
if [ "$ARM" = pages ]; then
    grep -a "latents: cached with" "$OUT/smoke.log" || { step "the smoke run trained on the cache's own stats"; exit 1; }
else
    grep -a "latents: as cached" "$OUT/smoke.log" || { step "the smoke run trained on other stats than the cache's"; exit 1; }
fi
# the transformer config NeMo consolidates must load as the glyph model's (twice the input and output channels)
smoke=$(ls -d "$OUT"/smoke_ckpt/epoch_*_step_* | sort -V | tail -1)
CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$smoke" --init "$INIT" --out "$OUT/model_smoke"
draw "$OUT/model_smoke" "$HELD_OUT" smoke 4 -n 1 --steps 8

step "train"
train train

step "the last checkpoint on the held-out windows and the held-in crops"
final=$(ls -d "$OUT"/epoch_*_step_* | sort -V | tail -1)
model="$OUT/model_final"
[ -d "$model" ] || CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$final" --init "$INIT" --out "$model"
draw "$model" "$HELD_OUT" final 4 -n 1000 --steps 50 || true  # a failed draw costs no other
draw "$model" "$HELD_OUT" final 1 -n 1000 --steps 50 || true
draw "$model" "$HELD_IN" heldin 4 -n 1000 --steps 50 || true

step "done"
touch "$REPO/runs/diffu_e2c_${ARM}_DONE"
