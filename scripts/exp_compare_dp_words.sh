#!/usr/bin/env bash
# diffu vs DiffusionPen at the WORD level (Protocol A — DiffusionPen's home turf, its own granularity + metrics).
# Both render the SAME IAM words; we measure legibility (a strong recognizer's CER on the generated word vs its
# intended text) and, optionally, FID vs real IAM words. This is the honesty check on raw image quality — if
# diffu loses here we say so; Protocol B (exp_compare_diffusionpen.sh) is the downstream-training metric.
#
#   GPU=2 DP_CMD='cd /home/blackwell/Desktop/DiffusionPen && ./.venv/bin/python gen_manifest.py --device cuda:2' scripts/exp_compare_dp_words.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
OUT=runs/htr_ab; CMP="$OUT/compare"
DIFFU=checkpoints/exp_sd35_128/step_120000/model.safetensors; ADAPTER=${ADAPTER:-runs/iam_lora_scaleup/final}
RECOG=${RECOG:-microsoft/trocr-base-handwritten}     # a strong INDEPENDENT reader (not one of our trained arms)
REAL_WORDS=${REAL_WORDS:-}                            # optional: a dir of real IAM word PNGs for FID
GPU=${GPU:-2}; N=${N:-5000}
DP_CMD=${DP_CMD:?set DP_CMD to the DiffusionPen gen_manifest.py invocation (isolated env)}
export CUDA_VISIBLE_DEVICES=$GPU
log(){ echo "[$(date +%T)] $*"; }

[ -f "$CMP/words.txt" ] || uv run python scripts/sample_iam_texts.py --n-words "$N" --out "$CMP"

log "render $N words with DIFFU"
[ -f "$CMP/diffu_words/train.jsonl" ] || CUDA_VISIBLE_DEVICES=$GPU uv run diffu-synth --ckpt "$DIFFU" --lora "$ADAPTER" \
  --hf-dataset Teklia/IAM-line --texts "$CMP/words.txt" --out "$CMP/diffu_words" >/dev/null 2>&1
log "render $N words with DIFFUSIONPEN"
[ -f "$CMP/dp_words/train.jsonl" ] || ( eval "$DP_CMD --texts $(pwd)/$CMP/words.txt --mode word --out $(pwd)/$CMP/dp_words" ) >/dev/null 2>&1

RES="$OUT/results_compare_dp_words.txt"
{ echo "=== WORD-LEVEL image quality: diffu vs DiffusionPen (same $N IAM words) ==="
  echo "-- legibility: $RECOG reads the generated word vs its intended text (lower CER = more legible) --"
  d=$(CUDA_VISIBLE_DEVICES=$GPU uv run diffu-eval --manifest "$CMP/diffu_words/train.jsonl" --htr "$RECOG" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}')
  p=$(CUDA_VISIBLE_DEVICES=$GPU uv run diffu-eval --manifest "$CMP/dp_words/train.jsonl"    --htr "$RECOG" 2>/dev/null | grep -oE 'CER [0-9.]+' | awk '{print $2}')
  printf "  diffu        legibility-CER %s\n  DiffusionPen legibility-CER %s\n" "$d" "$p"
} | tee "$RES"

# optional FID vs real IAM words (needs torch-fidelity + a real-words PNG dir)
if [ -n "$REAL_WORDS" ] && uv run python -c "import torch_fidelity" 2>/dev/null; then
  for m in diffu dp; do
    fid=$(CUDA_VISIBLE_DEVICES=$GPU uv run python -m torch_fidelity --fid --input1 "$CMP/${m}_words/lines" --input2 "$REAL_WORDS" 2>/dev/null | grep -i fid | grep -oE '[0-9.]+' | tail -1)
    echo "  $m FID-vs-real $fid" | tee -a "$RES"
  done
else
  echo "  (FID skipped — set REAL_WORDS=<dir of real IAM word pngs> and 'uv add torch-fidelity')" | tee -a "$RES"
fi
log "done -> $RES"
