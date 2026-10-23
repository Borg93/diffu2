#!/bin/bash
# v3c: v3's reference-style LoRA with references that show the same hand: a crop of the target's own page that shares
# none of its lines, or of its facing page (diffu_page.data.style_pairs --reference page -> data_flux3/style_pairs_v3c;
# 9,891 same-page and 354 facing-page pairs). v3's references came from any other page of the volume, often another
# scribe or another kind of document, so copying the reference's hand was often wrong in training. Also: a constant
# learning rate after warmup (v3's cosine decay stopped learning at ~800 steps), 3,000 steps, val_loss every 100 steps
# on 64 val-split pairs. Captions as v3's (no archive named). One GPU job at a time through jobs/spark_guard.sh:
#   1. cache the pairs and the validation pairs                         data_flux3/cache_style_v3c(_val)
#   2. a 5-step training smoke run
#   3. train the LoRA (v3's config with the overrides below)            runs/flux3_style_ref_lora_v3c
#   4. the final and middle checkpoints on the by-example set, the final on the held-out crops
#                                                                         runs/flux3_style_v3c/eval_*
# TRACKIO_TAG=<tag> names a rerun's trackio runs apart (flux3_v3c_lora_<tag>): trackio keeps a run per start, by name.
# Run detached:
#
#   setsid nohup jobs/flux3/run_v3c.sh >> runs/flux3_style_v3c/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/jobs/spark_guard.sh"
PAIRS="$REPO/data_flux3/style_pairs_v3c"
CACHE="$REPO/data_flux3/cache_style_v3c"
VAL_CACHE="$REPO/data_flux3/cache_style_v3c_val"  # 64 val-split pairs (style_pairs --val-pairs 64)
CONFIG="$REPO/packages/train/configs/flux3/flux3_style_ref_lora_v3.yaml"
OUT="$REPO/runs/flux3_style_v3c"
CKPT="$REPO/runs/flux3_style_ref_lora_v3c"
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

train() {  # train <name> [config overrides ...]
    local name=$1
    shift
    "$GUARD" --need-gb 60 --name "flux-v3c-$name" --log "$OUT/guard.log" -- \
        "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py -c "$CONFIG" --trackio.name "flux3_v3c_${name}${TRACKIO_TAG:+_$TRACKIO_TAG}" \
        --data.dataloader.cache_dir "$CACHE" "$@" \
        > "$OUT/$name.log" 2>&1
}

evaluate() {  # evaluate <name> <crops dir> <lora dir>
    local name=$1 crops=$2 lora=$3
    if "$GUARD" --need-gb 45 --name "flux-v3c-eval-$name" --log "$OUT/guard.log" -- \
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
    "$GUARD" --need-gb 30 --name flux-v3c-cache --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_pairs --manifest "$PAIRS/train.jsonl" \
        --out "$CACHE" --model "$MODEL" > "$OUT/cache.log" 2>&1
fi

if [ -f "$VAL_CACHE/metadata.json" ]; then
    echo "already cached: $VAL_CACHE"
else
    rm -rf "$VAL_CACHE"
    "$GUARD" --need-gb 30 --name flux-v3c-cache-val --log "$OUT/guard.log" -- \
        "$PY" -m diffu_train.cache_pairs --manifest "$PAIRS/val/val.jsonl" \
        --out "$VAL_CACHE" --model "$MODEL" > "$OUT/cache_val.log" 2>&1
fi

step "2/4 a 5-step smoke run"
rm -rf "$OUT/smoke_ckpt"  # NeMo resumes from a checkpoint it finds there: a rerun would train 0 steps
train smoke --step_scheduler.max_steps 5 --step_scheduler.ckpt_every_steps 1000 --lr_scheduler.lr_warmup_steps 1 \
    --checkpoint.checkpoint_dir "$OUT/smoke_ckpt"
grep -a "\[TRAIN\] step=" "$OUT/smoke.log" | tail -1
tail -1 "$OUT/guard.log"

step "3/4 train the LoRA"
train lora --checkpoint.checkpoint_dir "$CKPT" --step_scheduler.val_every_steps 100 \
    --step_scheduler.max_steps 3000 --step_scheduler.ckpt_every_steps 500 --lr_scheduler.lr_decay_style constant \
    --data.validation_dataloader._target_ \
    nemo_automodel.components.datasets.diffusion.image_edit_dataset.ImageEditDataloaderConfig \
    --data.validation_dataloader.cache_dir "$VAL_CACHE" --data.validation_dataloader.quantization 16 \
    --data.validation_dataloader.shuffle false --data.validation_dataloader.drop_last false \
    --data.validation_dataloader.num_workers 2

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
