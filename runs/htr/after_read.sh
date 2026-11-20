#!/bin/bash
# 20 Nov: GPU 2 after the read of pool_v2 (tmux `resume`), so it is not idle (GPU 2 only: GPU 1 and 0 are others').
#  1. The writing layer against the labels' outlines on 100,000 scans at random (CURATION E61).
#  2. Jev's score of the labelled lines of the first 20,000 of the same scans (E62; its first run with the model).
# Both resumable; stop with tmux kill-window -t htr:afterread.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'resume|parse2'; do sleep 60; done
log "the read has ended: $(wc -l < runs/htr/curate/pool_read/pool_v2_vllm.jsonl) pages"
CUDA_VISIBLE_DEVICES=2 uv run diffu-curate writing --out runs/htr/curate/writing --limit 100000 --workers 20 2>&1 | grep -v -i 'warn\|deprecat' | tail -n 3
log "writing layer done: $(ls runs/htr/curate/writing 2>/dev/null | wc -l) parts"
CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/vllm/.venv/bin/python -m diffu_htr_train.jev.score runs/htr/curate/jev/pairs.tsv runs/htr/curate/jev/scores --limit 20000 --memory 0.5 2>&1 | grep -v -i 'warn\|deprecat' | grep -E "scans|done|Error|Traceback" | tail -n 5
log "Jev done: $(cat runs/htr/curate/jev/scores/part-*.jsonl 2>/dev/null | wc -l) scans"
