#!/bin/bash
# 22 Nov only: GPU 0 is ours for today (Gabriel, 22 Nov). The writing layer per page (E61) runs there and is stopped
# at 23:50 at the latest; what is left goes on in tmux writingpages on GPU 2 once E60 has ended (it is resumable).
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
left=$(( $(date -d 'today 23:50' +%s) - $(date +%s) ))
[ "$left" -gt 60 ] || { log "past 23:50: not started"; exit 0; }
log "start on GPU 0, stopped after ${left} s at the latest"
CUDA_VISIBLE_DEVICES=0 timeout -s INT "$left" uv run diffu-curate writing --out /mnt/scratch/gabriel/htr/writing_pages --pages runs/htr/curate/train_pages.lance --workers 24 2>&1 | grep --line-buffered -v -i 'warn\|deprecat'
log "ended: $(ls /mnt/scratch/gabriel/htr/writing_pages | wc -l) parts"
