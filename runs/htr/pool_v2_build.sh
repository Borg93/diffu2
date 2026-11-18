#!/bin/bash
# 18 Nov (Gabriel away until Friday 20 Nov): the larger running-text set, built on the CPU when the folds are in.
# 541,000 scans at random of those stage_pool.py takes (an opening to cut by the head, no table, a fold line found,
# outside the ground truth's volumes): about 1 million single pages, about 615 GB under /mnt/scratch/gabriel/htr.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'placepool|foldjoin|poolcount'; do sleep 120; done
log "folds joined: staging"
OMP_NUM_THREADS=4 uv run python $P/stage_pool.py $S/pool_v2 541000 0 2>&1 | grep -v -i warn | tail -n 24 || { log "FAILED stage"; exit 1; }
log "building the pages"
nice -n 5 uv run diffu-reader-set --xml $S/pool_v2 --model nemotron-parse --cuts $S/pool_v2/cuts.json --split-from-gt-test-lists --workers 24 --out $S/pool_v2_nemotron/pages.jsonl 2>&1 | tail -n 4 || { log "FAILED set"; exit 1; }
wc -l < $S/pool_v2_nemotron/pages.jsonl > $S/pool_v2_nemotron/BUILT
log "set built: $(cat $S/pool_v2_nemotron/BUILT) pages"
