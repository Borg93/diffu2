#!/bin/bash
# GPU 1 after the embeddings (15 Nov): an andra arkiv pool sample built on CPU meanwhile, then read by Nemotron
# (nemotron-parse-more, guard 10) so every pool page read has its disagreement with its old label; stoppable any time.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
if [ ! -s $S/andra_pilot_nemotron/pages.jsonl ]; then
  uv run diffu-curate sample --pool andraarkiv --scans 34000 --seed 0 --out $S/andra_pilot || { log "FAILED sample"; exit 1; }
  uv run diffu-reader-set --xml $S/andra_pilot --model nemotron-parse --split-from-gt-test-lists --workers 16 --out $S/andra_pilot_nemotron/pages.jsonl || { log "FAILED set"; exit 1; }
fi
log "set: $(wc -l < $S/andra_pilot_nemotron/pages.jsonl) pages"
until grep -q '^done' runs/htr/curate/embed/egemma.log 2>/dev/null; do sleep 60; done
log "GPU 1 free of the embeddings: reading andra arkiv pool pages"
SET=$S/andra_pilot_nemotron/pages NO_REPEAT_NGRAM=10 CUDA_VISIBLE_DEVICES=1 packages/htr_train/envs/nemotron_parse/.venv/bin/python \
  runs/htr/verify/pool_read.py runs/htr/nemotron-parse-more/output/checkpoint-3094 0 1 $O/andra_pilot.jsonl > $O/andra_pilot.log 2>&1
log "read done: $(wc -l < $O/andra_pilot.jsonl)"
