#!/usr/bin/env bash
# LoRA capacity sweep for IAM adaptation, queued AFTER iam_lora_v2 (rank 32) finishes — GPU 2 is our only
# free GPU (0/1 hold the foreign vLLM job), so these run sequentially. All share the fixed recipe: init from
# the released base, warmup 150 (not the 2000 default), 8k steps, live gen_CER on. Compares rank 32/64/128.
#
#   GPU=2 scripts/exp_iam_lora_sweep.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
GPU=${GPU:-2}
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# wait for iam_lora_v2 (rank 32) to finish so we don't contend on the GPU
until [ -f runs/iam_lora_v2/final/pytorch_lora_weights.safetensors ]; do
  log "waiting for iam_lora_v2 to finish (GPU $GPU busy)"; sleep 120
done
log "iam_lora_v2 done — starting rank sweep on GPU $GPU"

run() {  # run <rank> <alpha> <out>
  local rank=$1 alpha=$2 out=$3
  [ -f "$out/final/pytorch_lora_weights.safetensors" ] && { log "skip $out (exists)"; return; }
  log "training $out  (rank=$rank alpha=$alpha)"
  CUDA_VISIBLE_DEVICES="$GPU" uv run diffu-train --init-from "$BASE" --no-repa \
    --lora --lora-rank "$rank" --lora-alpha "$alpha" --lr 3e-4 --warmup-steps 150 \
    --hf-dataset Teklia/IAM-line --hf-val-split validation --glyph-line --no-style-tokens \
    --htr-recognizer-eval microsoft/trocr-base-handwritten \
    --batch-size 8 --num-workers 4 --track --val-every 250 --sample-every 500 \
    --steps 8000 --save-every 4000 --out-dir "$out" > "$out.log" 2>&1
  log "done $out"
}

run 64  128 runs/iam_lora_v3_r64
log "sweep done — compare samples: runs/iam_lora_v{2,3_r64}/samples/step_008000.png"
