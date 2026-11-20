#!/bin/bash
# 20 Nov: after after_read.sh (the first 100,000 scans and Jev's trial), the writing layer goes on over every pool
# scan with labelled lines (3.07 million; resumable, in the same random order), then joins its columns. GPU 2 only.
# Stop it for a training run with tmux kill-window -t htr:afterread2; started again it goes on where it was.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'resume|parse2|afterread'; do sleep 60; done
log "the trial runs have ended: the writing layer over every scan"
CUDA_VISIBLE_DEVICES=2 uv run diffu-curate writing --out runs/htr/curate/writing --workers 24 2>&1 | grep -v -i 'warn\|deprecat' | tail -n 3
log "writing layer: $(ls runs/htr/curate/writing | wc -l) parts; joining"
flock runs/htr/curate/scans.lock uv run python - <<'PY'
import glob, pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from diffu_curate.table import add_features
features = pa.concat_tables([pq.read_table(part) for part in sorted(glob.glob("runs/htr/curate/writing/part-*.parquet"))])
print(f"{add_features(Path('runs/htr/curate/scans.lance'), features):,} of {len(features):,} scans given written, labelled, covered, filled")
PY
log "done"
