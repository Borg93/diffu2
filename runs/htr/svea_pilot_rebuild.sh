#!/bin/bash
# 14 Nov 18:35: the pilot set rebuilt with --split-from-gt-test-lists (the trainer refuses a real set without a split; the
# pool has no test lists and the sample leaves out every GT volume, so every page is a training page), then the chain.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr
log() { echo "[$(date '+%F %T')] $*"; }
uv run diffu-reader-set --xml $S/svea_pilot --model nemotron-parse --split-from-gt-test-lists --workers 32 --out $S/svea_pilot_nemotron/pages.jsonl || { log "FAILED set"; exit 1; }
log "set rebuilt: $(wc -l < $S/svea_pilot_nemotron/pages.jsonl) pages, splits: $(python3 -c "
import json,collections,sys
print(dict(collections.Counter(json.loads(l)['split'] for l in open(sys.argv[1]))))" $S/svea_pilot_nemotron/pages.meta.jsonl)"
ls $S/svea_pilot_nemotron/
bash runs/htr/svea_pilot.sh >> runs/htr/svea_pilot.log 2>&1
