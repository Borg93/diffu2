#!/usr/bin/env bash
# STAGE 1 of the scale-up: train a bigger diffu (GENERATOR) LoRA to make better/more-diverse fakes.
# The current adapter (runs/iam_lora, rank 32 / 4000 steps) already trains on ALL of IAM-line train; the
# only real "better" levers are rank + steps. This doubles both (rank 64, 8000 steps). Verify it makes good
# fakes (a quick gen + fidelity probe) BEFORE committing the ~8h 100k generation to it.
#
#   GPU=2 scripts/exp_diffu_lora_scaleup.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
GPU=${GPU:-2}; RANK=${RANK:-64}; ALPHA=${ALPHA:-128}; STEPS=${STEPS:-8000}
COMMON="--hf-dataset Teklia/IAM-line --hf-val-split validation --glyph-line --no-style-tokens"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$BASE" ] || { echo "missing base $BASE"; exit 1; }
log "training diffu-LoRA rank=$RANK alpha=$ALPHA steps=$STEPS on all IAM -> runs/iam_lora_scaleup"
CUDA_VISIBLE_DEVICES="$GPU" uv run diffu-train --init-from "$BASE" --no-repa \
  --lora --lora-rank "$RANK" --lora-alpha "$ALPHA" --lr 3e-4 --warmup-steps 200 \
  $COMMON --steps "$STEPS" --save-every 2000 --out-dir runs/iam_lora_scaleup --track
log "done -> runs/iam_lora_scaleup/final (verify with a quick diffu-synth + fidelity probe before 100k gen)"
