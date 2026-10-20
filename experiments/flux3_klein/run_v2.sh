#!/bin/bash
# The FLUX.3 Klein layout LoRA on the v2 crops (build_layout_crops.py -> data_flux3/layout_crops_v2): 40 pages
# per collection, captions that name the collection, the volume's years and the scan type (the style
# handles), margin notes, headings and page numbers with their roles, every crop at an exact NeMo bucket size.
# Waits for the pilot (run_pilot.sh) to finish, then one GPU job at a time, each through scripts/spark_guard.sh:
#   1. the base model on 24 held-out v2 crops, one per collection   runs/flux3_layout_v2/eval_base
#   2. cache latents + text embeddings of the v2 training crops      data_flux3/cache_train_v2
#   3. train the LoRA (flux3_layout_lora_v2.yaml)                     runs/flux3_layout_lora_v2
#   4. the v2 LoRA at its final and middle checkpoints, then the pilot's final LoRA, on the same 24 crops
# A failed eval is logged and skipped; failed caching or training stops the run. Run detached:
#
#   setsid nohup experiments/flux3_klein/run_v2.sh > runs/flux3_layout_v2/run.log 2>&1 < /dev/null &

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/scripts/spark_guard.sh"
CROPS="$REPO/data_flux3/layout_crops_v2"
OUT="$REPO/runs/flux3_layout_v2"
CKPT="$REPO/runs/flux3_layout_lora_v2"
PILOT_CKPT="$REPO/runs/flux3_layout_lora"
N_EVAL=24
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's tools/ and examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

evaluate() {  # evaluate <name> [lora dir]
    local name=$1 lora=${2:-}
    if "$GUARD" --need-gb 40 --name "flux-v2-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" "$REPO/experiments/flux3_klein/eval_layout.py" --crops "$CROPS/val_eval" \
        -n "$N_EVAL" --steps 50 --out "$OUT/eval_$name" ${lora:+--lora "$lora"} > "$OUT/eval_$name.log" 2>&1; then
        tail -1 "$OUT/eval_$name.log"
    else
        echo "eval $name failed, see $OUT/eval_$name.log"
    fi
}

step "waiting for the pilot to finish"
while pgrep -f "flux3_klein/run_pilo[t]" > /dev/null; do sleep 60; done

step "0/4 pick $N_EVAL held-out crops, one per collection"
"$REPO/.venv/bin/python" "$REPO/experiments/flux3_klein/pick_eval_crops.py" "$CROPS/val" "$CROPS/val_eval" -n "$N_EVAL"

step "1/4 base model on the held-out crops"
evaluate base

step "2/4 cache the v2 training crops"
rm -rf "$REPO/data_flux3/cache_train_v2"
"$GUARD" --need-gb 25 --name flux-v2-cache --log "$OUT/guard.log" -- \
    "$PY" -m tools.diffusion.preprocessing_multiprocess image --processor flux3 --model_name "$MODEL" \
    --dataset_name "$CROPS/train" --dataset_media_column image \
    --dataset_caption_column caption --resolution_preset 1024p --output_dir "$REPO/data_flux3/cache_train_v2" \
    > "$OUT/cache.log" 2>&1

step "3/4 train the LoRA"
# bucket_sampler.py (the config's dataloader) lives next to this script
PYTHONPATH="$REPO/experiments/flux3_klein${PYTHONPATH:+:$PYTHONPATH}" \
    "$GUARD" --need-gb 45 --name flux-v2-lora --log "$OUT/guard.log" -- \
    "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py \
    -c "$REPO/experiments/flux3_klein/flux3_layout_lora_v2.yaml" > "$OUT/train.log" 2>&1

step "4/4 the v2 LoRA at its final and middle checkpoints, then the pilot's final LoRA"
mapfile -t ckpts < <(ls -d "$CKPT"/epoch_*_step_* | sort -V)
middle=${ckpts[$(( ${#ckpts[@]} / 2 ))]}
final=${ckpts[-1]}
evaluate "step_${final##*_}" "$final/model"
evaluate "step_${middle##*_}" "$middle/model"
pilot=$(ls -d "$PILOT_CKPT"/epoch_*_step_* | sort -V | tail -1)
evaluate "pilot_step_${pilot##*_}" "$pilot/model"

step "done"
touch "$OUT/DONE"
