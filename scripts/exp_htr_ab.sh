#!/usr/bin/env bash
# Downstream augmentation A/B: does diffu's synthetic data work as HTR training data?
#   real IAM  ->  TrOCR(real)         --\
#   synth (from a diffu checkpoint)     >-- eval CER on held-out REAL IAM test  -> does synth help?
#   real+synth -> TrOCR(real+synth)   --/
# TrOCR fine-tunes on accelerate (HF Seq2SeqTrainer) and logs to trackio (project 'diffu-htr').
# Queued: waits for the diffu LoRA adapter AND a free GPU (the diffu arms hold GPU 1/2), then runs.
#
#   scripts/exp_htr_ab.sh [adapter_dir]     # default runs/iam_lora/final
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."

BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
ADAPTER=${1:-runs/iam_lora/final}
HTRBASE=${HTRBASE:-microsoft/trocr-base-handwritten}   # English HTR base (IAM is English)
N_SYNTH=${N_SYNTH:-4000}; N_REAL=${N_REAL:-2000}; N_TEST=${N_TEST:-1000}; EPOCHS=${EPOCHS:-5}
OUT=runs/htr_ab; mkdir -p "$OUT"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1) wait for the diffu LoRA adapter
until [ -f "$ADAPTER/pytorch_lora_weights.safetensors" ]; do log "waiting for adapter $ADAPTER"; sleep 120; done
log "adapter ready: $ADAPTER"

# 2) wait for a free GPU among 1,2 (a diffu arm releases one when it finishes)
GPU=""
while [ -z "$GPU" ]; do
  for g in 1 2; do
    m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 99999)
    if [ "$m" -lt 3000 ]; then GPU=$g; break; fi
  done
  [ -z "$GPU" ] && { log "waiting for a free GPU (1/2)"; sleep 120; }
done
log "running HTR A/B on GPU $GPU"
export CUDA_VISIBLE_DEVICES=$GPU

# 3) materialize REAL IAM (train subset + test) as local manifests
uv run python -c "from diffu.data.hf_dataset import hf_to_manifest as h; \
  h('Teklia/IAM-line','train','$OUT/real',max_rows=$N_REAL); h('Teklia/IAM-line','test','$OUT/test',max_rows=$N_TEST)"

# 4) generate SYNTH lines from the diffu checkpoint (+adapter)
uv run diffu-synth --ckpt "$BASE" --lora "$ADAPTER" --hf-dataset Teklia/IAM-line --split train \
  --n "$N_SYNTH" --out "$OUT/synth"
cat "$OUT/real/train.jsonl" "$OUT/synth/train.jsonl" > "$OUT/real_plus_synth.jsonl"

# 5) fine-tune TrOCR: real vs real+synth (accelerate + trackio)
uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/real/train.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_real" --epochs "$EPOCHS" --track
uv run diffu-htr-train --base "$HTRBASE" --train "$OUT/real_plus_synth.jsonl" --val "$OUT/test/test.jsonl" \
  --out "$OUT/htr_realsynth" --epochs "$EPOCHS" --track

# 6) evaluate both on held-out REAL IAM test -> CER (synth helps iff realsynth CER < real CER)
{ echo "=== CER on real IAM test (base=$HTRBASE, real=$N_REAL synth=$N_SYNTH) ==="
  echo "-- TrOCR(real):";       uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_real"
  echo "-- TrOCR(real+synth):"; uv run diffu-eval --manifest "$OUT/test/test.jsonl" --htr "$OUT/htr_realsynth"
} 2>&1 | tee "$OUT/results.txt"
log "done -> $OUT/results.txt"
