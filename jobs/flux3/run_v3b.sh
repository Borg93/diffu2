#!/bin/bash
# v3b: v3's reference-style LoRA with the style in words as well. The same targets and references as v3; half the
# captions also name the target's archive, years, paper and ink (diffu_page.data.style_pairs --named-fraction 0.5 ->
# data_flux3/style_pairs_v3b). Its by-example test names the template's own archive in words while the reference
# comes from another archive: a hand that follows the reference shows the picture wins over the words. One GPU job at
# a time, each through jobs/spark_guard.sh:
#   1. cache the pairs with NeMo's FLUX.3 image-edit encoder             data_flux3/cache_style_v3b
#   2. a 5-step training smoke run: fails fast, and its log shows memory and step time
#   3. train the LoRA (v3's config, its cache and checkpoint dirs)      runs/flux3_style_ref_lora_v3b
#   4. the final checkpoint on the by-example set (+ diffu_page.cli.sheets) and the held-out crops, the middle
#      checkpoint on the by-example set                                   runs/flux3_style_v3b/eval_*
# A failed eval is logged and skipped; failed caching or training stops the run. A rerun skips a finished cache.
# Run detached:
#
#   setsid nohup jobs/flux3/run_v3b.sh >> runs/flux3_style_v3b/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/jobs/spark_guard.sh"
PAIRS="$REPO/data_flux3/style_pairs_v3b"
CACHE="$REPO/data_flux3/cache_style_v3b"
CONFIG="$REPO/packages/train/configs/flux3/flux3_style_ref_lora_v3.yaml"
OUT="$REPO/runs/flux3_style_v3b"
CKPT="$REPO/runs/flux3_style_ref_lora_v3b"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 60 --name "flux-v3b-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" --trackio.name "flux3_v3b_$name" \
        --data.dataloader.cache_dir "$CACHE" "$@" \
        > "$OUT/$name.log" 2>&1
}

evaluate() {  # evaluate <name> <crops dir> <lora dir>
    local name=$1 crops=$2 lora=$3
    if "$GUARD" --need-gb 45 --name "flux-v3b-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" -m diffu_page.cli.evaluate --crops "$crops" -n 1000 --steps 50 \
        --out "$OUT/eval_$name" --lora "v3=$lora" > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

step "1/4 cache the pairs"
# the encoder writes the merged metadata.json last, so it marks a finished cache
if [ -f "$CACHE/metadata.json" ]; then
    echo "already cached: $CACHE"
else
    rm -rf "$CACHE"
    "$GUARD" --need-gb 30 --name flux-v3b-cache --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_pairs --manifest "$PAIRS/train.jsonl" \
        --out "$CACHE" --model "$MODEL" > "$OUT/cache.log" 2>&1
fi

step "2/4 a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=" "$OUT/smoke.log" | tail -1
tail -1 "$OUT/guard.log"

step "3/4 train the LoRA"
train lora --checkpoint.checkpoint_dir "$CKPT"

step "4/4 evaluate the final and middle checkpoints"
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
