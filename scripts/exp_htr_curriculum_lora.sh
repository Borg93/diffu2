#!/usr/bin/env bash
# The --lora arm of the curriculum: reuse the synth-pretrained TrOCR (htr_synth_pretrain), then LoRA-adapt
# on the real domain (parameter-efficient) instead of full FT. Queued after the full-FT curriculum frees
# GPU 2. Compares to real-only (9.50%), best mix (0.25:1, 9.01%), and curriculum-full.
#
#   GPU=2 scripts/exp_htr_curriculum_lora.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; GPU=${GPU:-2}; export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# wait for the full-FT curriculum to finish (frees the GPU AND guarantees htr_synth_pretrain exists)
until [ -f "$OUT/htr_curriculum_full/model.safetensors" ]; do log "waiting for the full-FT curriculum"; sleep 120; done
[ -f "$OUT/htr_synth_pretrain/model.safetensors" ] || { log "missing synth-pretrain"; exit 1; }

log "synth-pretrain ready — LoRA-adapting on 2000 real (decoder attention, rank 16)"
uv run diffu-htr-train --base "$OUT/htr_synth_pretrain" --train "$OUT/real/train.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_curriculum_lora" --lora --lora-rank 16 --epochs 5 --track

{ echo "=== CURRICULUM-LoRA (synth-pretrain -> LoRA on real) — CER on real IAM test ==="
  echo "  references: real-only 9.50% | best MIX (0.25:1) 9.01% | curriculum-full see results_curriculum.txt"
  uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_curriculum_lora"
} 2>&1 | tee "$OUT/results_curriculum_lora.txt"
log "curriculum-LoRA done -> $OUT/results_curriculum_lora.txt"
