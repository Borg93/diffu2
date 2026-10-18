#!/usr/bin/env bash
# IAM adaptation experiment — three trackio-logged arms on Teklia/IAM-line.
#   A) LoRA fine-tune from the released Swedish base (step_120000)
#   B) full fine-tune from the same base
#   C) from scratch on IAM (the DiffusionPen-comparable, IAM-native arm)
# All arms share the base architecture flags (--glyph-line --no-style-tokens) so --init-from loads,
# and log to trackio project "diffu" (run name = out-dir basename). View with:  uv run trackio show
#
# Usage:
#   scripts/exp_iam.sh launch      # fire all arms detached (A->B on GPU1, C on GPU2); survives logout
#   scripts/exp_iam.sh lora  <gpu> # run one arm in the foreground
#   scripts/exp_iam.sh full  <gpu>
#   scripts/exp_iam.sh scratch <gpu>
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."

BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
COMMON="--hf-dataset Teklia/IAM-line --hf-val-split validation --glyph-line --no-style-tokens \
        --batch-size 8 --num-workers 4 --no-cer --track --val-every 250 --sample-every 500"
mkdir -p runs

lora() {    # ~50 MB adapter; fastest arm
  # --warmup-steps 150: the default (2000) is for the 60k scratch run; on a 4k-step LoRA it makes HALF
  # the run warmup -> the adapter never trains at sustained LR (undertrained). Short warmup for short runs.
  CUDA_VISIBLE_DEVICES="$1" uv run diffu-train --init-from "$BASE" --no-repa \
    --lora --lora-rank 32 --lora-alpha 64 --lr 3e-4 --warmup-steps 150 \
    $COMMON --steps 4000 --save-every 1000 --out-dir runs/iam_lora
}
full() {    # full backbone fine-tune from base (17 GB checkpoints -> save rarely)
  CUDA_VISIBLE_DEVICES="$1" uv run diffu-train --init-from "$BASE" --no-repa \
    --lr 1e-5 --warmup-steps 250 $COMMON --steps 6000 --save-every 3000 --out-dir runs/iam_full
}
scratch() { # from random init; REPA on (helps early); many hours
  CUDA_VISIBLE_DEVICES="$1" uv run diffu-train \
    --lr 1e-4 --warmup-steps 2000 $COMMON --steps 60000 --save-every 10000 --out-dir runs/iam_scratch
}

case "${1:-launch}" in
  lora)    lora "${2:-1}";;
  full)    full "${2:-1}";;
  scratch) scratch "${2:-2}";;
  launch)
    # GPU 1: LoRA then full (sequential). GPU 2: from scratch. Detached (setsid) + logged; survives exit.
    setsid bash -c "bash '$SELF' lora 1 >runs/iam_lora.log 2>&1; bash '$SELF' full 1 >runs/iam_full.log 2>&1" \
      >/dev/null 2>&1 < /dev/null &
    echo "GPU1: iam_lora -> iam_full   (logs: runs/iam_lora.log, runs/iam_full.log)"
    setsid bash -c "bash '$SELF' scratch 2 >runs/iam_scratch.log 2>&1" >/dev/null 2>&1 < /dev/null &
    echo "GPU2: iam_scratch            (log: runs/iam_scratch.log)"
    echo "track: uv run trackio show   (project 'diffu': iam_lora / iam_full / iam_scratch)"
    ;;
  *) echo "usage: $0 {launch|lora|full|scratch} [gpu]"; exit 1;;
esac
