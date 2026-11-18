#!/bin/bash
# 18 Nov: CPU work for the absence until Friday. Orientation (E58) of every page of the trusted set now, and of the
# million pages of pool_v2 when they are built. Resumable; nice, beside the training and the reads.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes; O=runs/htr/curate/orientation; mkdir -p $O
log() { echo "[$(date '+%F %T')] $*"; }
WORKERS=12 THREADS=4 nice -n 15 uv run python $P/orientation_pages.py $S/trusted_v1_nemotron/pages $O/trusted_v1.parquet 2>&1 | grep -v -i warn | tail -n 2
log "trusted pages done"
until [ -s $S/pool_v2_nemotron/BUILT ]; do sleep 300; tmux list-windows -t htr -F '#W' | grep -q -x poolv2build || { [ -s $S/pool_v2_nemotron/BUILT ] || { log "pool_v2 was not built: stopping"; exit 1; }; }; done
WORKERS=32 THREADS=8 nice -n 15 uv run python $P/orientation_pages.py $S/pool_v2_nemotron/pages $O/pool_v2.parquet 2>&1 | grep -v -i warn | tail -n 2
log "pool_v2 pages done"
