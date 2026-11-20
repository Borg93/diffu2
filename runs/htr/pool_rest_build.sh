#!/bin/bash
# 20 Nov (Gabriel: yes): the running-text pages of the pools that pool_v2 does not hold, built on the CPU in three sets
# of about 587,000 scans (about 1.08 million pages, about 670 GB each) so each is usable when done. Resumable by set.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes
log() { echo "[$(date '+%F %T')] $*"; }
EX=$S/pool_v2/scans.parquet
for n in 3 4 5; do
  if [ ! -s $S/pool_v${n}_nemotron/BUILT ]; then
    free=$(df --output=avail -BG /mnt/scratch | tail -n 1 | tr -dc 0-9); [ "$free" -gt 900 ] || { log "only $free GB free on scratch: stopping before pool_v$n"; exit 1; }
    [ -s $S/pool_v$n/scans.parquet ] || EXCLUDE=$EX OMP_NUM_THREADS=4 uv run python $P/stage_pool.py $S/pool_v$n 587200 $n 2>&1 | grep -v -i warn | tail -n 2 || { log "FAILED stage $n"; exit 1; }
    log "pool_v$n staged: building the pages"
    nice -n 10 uv run diffu-reader-set --xml $S/pool_v$n --model nemotron-parse --cuts $S/pool_v$n/cuts.json --split-from-gt-test-lists --workers 20 --out $S/pool_v${n}_nemotron/pages.jsonl 2>&1 | tail -n 3 || { log "FAILED set $n"; exit 1; }
    wc -l < $S/pool_v${n}_nemotron/pages.jsonl > $S/pool_v${n}_nemotron/BUILT; log "pool_v$n built: $(cat $S/pool_v${n}_nemotron/BUILT) pages"
  fi
  EX=$EX,$S/pool_v$n/scans.parquet
done
log "done"
