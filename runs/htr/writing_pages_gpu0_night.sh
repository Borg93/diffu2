#!/bin/bash
# 22 Nov: Gabriel gave GPU 0 until 08:00 on 23 Nov. When the evening run (stopped at 23:50) has ended, the per-page
# writing layer goes on there and is stopped at 07:50 at the latest; what is left then runs on GPU 2 after E60
# (tmux writingpages). Resumable.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
while tmux has-session -t writingpages0 2>/dev/null; do sleep 30; done
left=$(( $(date -d '2026-11-23 07:50' +%s) - $(date +%s) ))
[ "$left" -gt 60 ] || { log "past 07:50: not started"; exit 0; }
log "start on GPU 0, stopped after ${left} s at the latest"
CUDA_VISIBLE_DEVICES=0 timeout -s INT "$left" uv run diffu-curate writing --out /mnt/scratch/gabriel/htr/writing_pages --pages runs/htr/curate/train_pages.lance --workers 24 2>&1 | grep --line-buffered -v -i 'warn\|deprecat'
log "ended: $(ls /mnt/scratch/gabriel/htr/writing_pages | wc -l) parts"
