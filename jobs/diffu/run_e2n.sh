#!/bin/bash
# E2n: E2b with our pages' latent stats. Under Qwen-Image's stats our pages' latents have 8.3 % of unit variance, so
# every diffu run so far trained at about 1/12 the signal-to-noise ratio its timestep schedule assumes (TODO.md, diffu).
# Measures the stats over E2b's cache (diffu_train.latent_stats), writes diffu-S from E2b's own start
# (runs/diffu_s_vl_init's weights) with its own VAE normalizing with them (diffu-init --latent-stats), and trains it on
# E2b's cache converted on load, with E2b's settings and learning-rate curve, for 5,000 of E2b's 10,000 steps
# (packages/train/configs/diffu/diffu_s_e2n_glyph.yaml). Its last checkpoint (the only one NeMo consolidates) is drawn on
# the held-out windows (data_flux3/e2_val_eval/glyphs) and the 24 held-in crops (data_flux3/e2_train_eval/glyphs), to
# set beside E2b's step 5,000. One GPU job at a time through jobs/spark_guard.sh. Run detached:
#
#   mkdir -p runs/diffu_e2n && setsid nohup jobs/diffu/run_e2n.sh >> runs/diffu_e2n/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
GUARD="$REPO/jobs/spark_guard.sh"
START="$REPO/runs/diffu_s_vl_init"
CACHE="$REPO/data_flux3/cache_diffu_e2b_glyph"
INIT="$REPO/runs/diffu_s_vl_pages_init"
CONFIG="$REPO/packages/train/configs/diffu/diffu_s_e2n_glyph.yaml"
OUT="$REPO/runs/diffu_s_e2n_glyph"
HELD_OUT="$REPO/data_flux3/e2_val_eval/glyphs"
HELD_IN="$REPO/data_flux3/e2_train_eval/glyphs"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" DIFFU_INIT="$INIT" DIFFU_CACHE="$CACHE"

[ -f "$CACHE/metadata.json" ] || { echo "no finished cache at $CACHE: run jobs/diffu/run_e2b.sh first"; exit 1; }
[ -f "$START/modular_model_index.json" ] || { echo "no E2b start at $START: run jobs/diffu/run_e2b.sh first"; exit 1; }
for set_dir in "$HELD_OUT" "$HELD_IN"; do
    [ -f "$set_dir/metadata.jsonl" ] || { echo "no eval crops at $set_dir"; exit 1; }
done
step() { echo "=== $(date '+%F %T') $*"; }
mkdir -p "$OUT"

step "our pages' latent stats over the cache's crops (diffu_train.latent_stats)"
[ -f "$OUT/latent_stats.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/python" -m diffu_train.latent_stats --cache "$CACHE" \
        --out "$OUT/latent_stats.json"

step "diffu-S from E2b's start, its own VAE normalizing with our pages' stats"
[ -f "$INIT/modular_model_index.json" ] ||
    CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-init" --size S --out "$INIT" --weights-from "$START" \
        --latent-stats "$OUT/latent_stats.json"

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    (cd "$AUTOMODEL" && "$GUARD" --need-gb 40 --name "diffu-e2n-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" \
        --trackio.name "diffu_s_e2n_glyph_$name" "$@" > "$OUT/$name.log" 2>&1)
}

step "a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "caption dropout: 0.1 " "$OUT/smoke.log" || { step "the smoke run trained without caption dropout"; exit 1; }
grep -a "latents: cached with" "$OUT/smoke.log" || { step "the smoke run trained on the cache's own stats"; exit 1; }

step "train"
train train

step "the last checkpoint on the held-out windows and the held-in crops"
final=$(ls -d "$OUT"/epoch_*_step_* | sort -V | tail -1)
model="$OUT/model_final"
[ -d "$model" ] || CUDA_VISIBLE_DEVICES="" "$REPO/.venv/bin/diffu-export" --checkpoint "$final" --init "$INIT" --out "$model"

draw() {  # draw <eval set> <name> <guidance>
    if "$GUARD" --need-gb 30 --name "diffu-e2n-eval-$2-g$3" --log "$OUT/guard.log" -- \
        "$REPO/.venv/bin/python" -m diffu_page.cli.evaluate --crops "$1" --model "$model" -n 1000 \
        --steps 50 --guidance "$3" --out "$OUT/eval_$2_g$3" > "$OUT/eval_$2_g$3.log" 2>&1; then
        tail -1 "$OUT/eval_$2_g$3.log"
    else
        step "eval of $2 at guidance $3 FAILED, see $OUT/eval_$2_g$3.log"
    fi
}
draw "$HELD_OUT" final 4
draw "$HELD_OUT" final 1
draw "$HELD_IN" heldin 4

step "done"
touch "$REPO/runs/diffu_e2n_DONE"
