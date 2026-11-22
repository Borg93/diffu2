#!/bin/bash
# 22 Nov: the writing layer per page (E61) over the scans of train_pages.lance, on GPU 2 once E60 has ended (beside
# the training it did 14 scans a second and slowed the training from 22 to 33 s a step; alone about 70).
# Resumable: started again it goes on where it was. Stop it with tmux kill-session -t writingpages.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x afterwriting; do sleep 120; done
log "E60 has ended: start"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 uv run diffu-curate writing --out /mnt/scratch/gabriel/htr/writing_pages --pages runs/htr/curate/train_pages.lance --workers 24 2>&1 | grep --line-buffered -v -i 'warn\|deprecat'
log "done: $(ls /mnt/scratch/gabriel/htr/writing_pages | wc -l) parts"
