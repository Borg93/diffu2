#!/usr/bin/env bash
# STAGE 2: bulk-generate ~100k IAM fakes with the chosen diffu-LoRA, for the pretrain-then-finetune scale-up.
# Chunked (10 x 10k, distinct seeds) so it is RESUMABLE — synth.py truncates its manifest on restart, so a
# whole-100k single job would lose everything on interruption; a finished 10k chunk is skipped on rerun and
# at most one in-flight chunk is ever redone. Pools all chunks into synth100k/train.jsonl at the end.
#
#   ADAPTER=runs/iam_lora_scaleup/final GPU=2 scripts/exp_gen_100k.sh
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)/.."
BASE=checkpoints/exp_sd35_128/step_120000/model.safetensors
ADAPTER=${ADAPTER:?set ADAPTER to the chosen diffu-LoRA (e.g. runs/iam_lora_scaleup/final)}
GPU=${GPU:-2}; CHUNK=${CHUNK:-10000}; NCHUNKS=${NCHUNKS:-10}
CHUNKS=${CHUNKS:-$(seq 1 "$NCHUNKS")}   # which chunk indices THIS job generates (split across GPUs)
POOL_AT_END=${POOL_AT_END:-1}           # set 0 when several GPUs run in parallel — let the orchestrator pool
OUT=runs/htr_ab/synth100k
log(){ echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p "$OUT"

[ -f "$BASE" ] || { echo "missing base $BASE"; exit 1; }
[ -d "$ADAPTER" ] || { echo "missing adapter $ADAPTER"; exit 1; }
log "generating chunks [$CHUNKS] on GPU $GPU with adapter $ADAPTER (chunk size $CHUNK)"

for c in $CHUNKS; do
  cdir="$OUT/chunk_$c"
  if [ -f "$cdir/train.jsonl" ] && [ "$(wc -l < "$cdir/train.jsonl")" -ge "$CHUNK" ]; then
    log "chunk $c already complete ($(wc -l < "$cdir/train.jsonl") lines) — skip"; continue
  fi
  log "chunk $c (seed $((10+c)))"
  CUDA_VISIBLE_DEVICES="$GPU" uv run diffu-synth --ckpt "$BASE" --lora "$ADAPTER" \
    --hf-dataset Teklia/IAM-line --split train --n "$CHUNK" --seed "$((10+c))" --out "$cdir"
done

# Pool only when ALL NCHUNKS chunks are complete — so whichever GPU finishes last does the pooling.
done_chunks=0
for c in $(seq 1 "$NCHUNKS"); do
  if [ -f "$OUT/chunk_$c/train.jsonl" ] && [ "$(wc -l < "$OUT/chunk_$c/train.jsonl")" -ge "$CHUNK" ]; then
    done_chunks=$((done_chunks+1))
  fi
done
if [ "$POOL_AT_END" = 1 ] && [ "$done_chunks" -ge "$NCHUNKS" ]; then
  log "all $NCHUNKS chunks complete — pooling -> $OUT/train.jsonl"
  cat "$OUT"/chunk_*/train.jsonl > "$OUT/train.jsonl"
  log "done: $(wc -l < "$OUT/train.jsonl") fakes -> $OUT/train.jsonl"
else
  log "my chunks done ($done_chunks/$NCHUNKS total complete); pooling left to the orchestrator"
fi
