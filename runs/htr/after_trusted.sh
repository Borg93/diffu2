#!/bin/bash
# 17 Nov (CURATION E40): once the trusted-cut training queue has ended (its tmux window `trusted` is gone), every pool
# scan with labelled lines is embedded by DINOv3 and TIPSv2, one shard a GPU, and every candidate opening gets its
# fold placed on the CPU, so that the cut rule can be applied to the whole pools. Nothing here trains. Both jobs are
# resumable: start this script again after a stop.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x trusted; do sleep 60; done
log "the training queue has ended"
grep -q "all done" runs/htr/trusted_pretrain.log || log "the training queue did not reach its end: see runs/htr/trusted_pretrain.log"
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s + 1)) uv run --with sentencepiece python $P/embed_pool.py $S/embed_pool $s 2 > runs/htr/curate/embed_pool_$s.log 2>&1 &
done
PYTHONPATH=$P nice -n 10 uv run python $P/place_folds.py 16 pool > runs/htr/curate/place_folds_pool2.log 2>&1 &
wait
log "ended: $(ls $S/embed_pool 2> /dev/null | wc -l) parts of vectors; $(tail -n 1 runs/htr/curate/embed_pool_0.log) | $(tail -n 1 runs/htr/curate/embed_pool_1.log) | folds: $(tail -n 1 runs/htr/curate/place_folds_pool2.log)"
