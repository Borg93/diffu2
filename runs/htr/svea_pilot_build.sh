#!/bin/bash
# 14 Nov: the control experiment's pretraining set: 34,000 random Svea pool scans (old htrflow labels as they are, GT
# volumes left out), written as Nemotron 3.5 Parse reads a page.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr
log() { echo "[$(date '+%F %T')] $*"; }
uv run python packages/htr_train/lion_data_prep/curate/pilot_sample.py --scans 34000 --seed 0 --out $S/svea_pilot || { log "FAILED sample"; exit 1; }
log "sampled"
uv run diffu-reader-set --xml $S/svea_pilot --model nemotron-parse --workers 24 --out $S/svea_pilot_nemotron/pages.jsonl || { log "FAILED set"; exit 1; }
log "set built: $(wc -l < $S/svea_pilot_nemotron/pages.jsonl) pages"
