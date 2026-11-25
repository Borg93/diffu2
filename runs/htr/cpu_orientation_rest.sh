#!/bin/bash
# 23 Nov: which way up the writing is on every page of pool_v3 to pool_v5 (3.2 million pages), on the CPU beside the
# fine-tune on GPU 2 (nice, 24 workers: the trainer's loaders keep their cores). Resumable.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; O=runs/htr/curate/orientation; P=runs/htr/curate/prototypes
for n in pool_v3 pool_v4 pool_v5; do
    echo "[$(date '+%F %T')] $n"
    WORKERS=24 THREADS=4 nice -n 15 uv run python $P/orientation_pages.py $S/${n}_nemotron/pages $O/$n.parquet 2>&1 | grep --line-buffered -v -i warn | tail -n 2
done
echo "[$(date '+%F %T')] done"
