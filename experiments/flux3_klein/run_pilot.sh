#!/bin/bash
# The FLUX.3 Klein layout-LoRA pilot, end to end, one GPU job at a time (each through scripts/spark_guard.sh):
#   1. the base model on held-out crops (the baseline)        runs/flux3_layout/eval_base
#   2. cache latents + text embeddings of the training crops  data_flux3/cache_train
#   3. train the LoRA (flux3_layout_lora.yaml)                 runs/flux3_layout_lora
#   4. the LoRA at its middle and final checkpoints            runs/flux3_layout/eval_step_*
# Stops at the first failing step. Run detached so it outlives the shell:
#
#   setsid nohup experiments/flux3_klein/run_pilot.sh > runs/flux3_layout/pilot.log 2>&1 < /dev/null &
#
# Needs ~/Desktop/repos/Automodel/.venv-managed (NeMo AutoModel + diffusers main on a uv-managed Python, which
# ships the headers Triton compiles against) and the model at ~/Desktop/repos/flux3-models/flux-3-image-klein-base.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
AUTOMODEL="$HOME/Desktop/repos/Automodel"
PY="$AUTOMODEL/.venv-managed/bin/python"
TORCHRUN="$AUTOMODEL/.venv-managed/bin/torchrun"
MODEL="$HOME/Desktop/repos/flux3-models/flux-3-image-klein-base"
GUARD="$REPO/scripts/spark_guard.sh"
OUT="$REPO/runs/flux3_layout"
CKPT="$REPO/runs/flux3_layout_lora"
N_EVAL=24
export CUDA_VISIBLE_DEVICES=0
export DIFFU_REPO="$REPO" FLUX3_MODEL="$MODEL"  # the training config's paths

mkdir -p "$OUT"
cd "$AUTOMODEL"  # NeMo's tools/ and examples/ are run from its repo root
step() { echo "=== $(date '+%F %T') $*"; }

evaluate() {  # evaluate <name> [lora dir]
    local name=$1 lora=${2:-}
    "$GUARD" --need-gb 40 --name "flux-eval-$name" --log "$OUT/guard.log" -- \
        "$PY" "$REPO/experiments/flux3_klein/eval_layout.py" --crops "$REPO/data_flux3/layout_crops/val" \
        -n "$N_EVAL" --steps 50 --out "$OUT/eval_$name" ${lora:+--lora "$lora"} > "$OUT/eval_$name.log" 2>&1
    tail -1 "$OUT/eval_$name.log"
}

step "1/4 base model on $N_EVAL held-out crops"
evaluate base

step "2/4 cache the training crops"
rm -rf "$REPO/data_flux3/cache_train"
"$GUARD" --need-gb 25 --name flux-cache --log "$OUT/guard.log" -- \
    "$PY" -m tools.diffusion.preprocessing_multiprocess image --processor flux3 --model_name "$MODEL" \
    --dataset_name "$REPO/data_flux3/layout_crops/train" --dataset_media_column image \
    --dataset_caption_column caption --resolution_preset 1024p --output_dir "$REPO/data_flux3/cache_train" \
    > "$OUT/cache.log" 2>&1

step "3/4 train the LoRA"
"$GUARD" --need-gb 45 --name flux-lora --log "$OUT/guard.log" -- \
    "$TORCHRUN" --nproc-per-node=1 examples/diffusion/finetune/finetune.py \
    -c "$REPO/experiments/flux3_klein/flux3_layout_lora.yaml" > "$OUT/train.log" 2>&1

step "4/4 the LoRA at its middle and final checkpoints"
mapfile -t ckpts < <(ls -d "$CKPT"/epoch_*_step_* | sort -V)
middle=${ckpts[$(( ${#ckpts[@]} / 2 ))]}
final=${ckpts[-1]}
evaluate "step_${middle##*_}" "$middle/model"
evaluate "step_${final##*_}" "$final/model"

step "done"
touch "$OUT/DONE"
