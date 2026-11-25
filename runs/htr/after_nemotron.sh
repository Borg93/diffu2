#!/bin/bash
# After the Nemotron 3.5 Parse run (13 Nov, the user's yes): its best checkpoint reads the 200 test pages Unlimited-OCR 2
# and LightOnOCR-3 read (both GPUs, 3 h at most), its CER scored the same way.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q -E "nemotron-parse-all (finished|stopped [0-9]+ times more)" runs/htr/after_lightonocr3.log; do sleep 120; done
log "nemotron-parse-all ended: $(grep -E 'nemotron-parse-all (finished|stopped)' runs/htr/after_lightonocr3.log | tail -1)"
R=runs/htr/nemotron-parse-all
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'] or '')" \
  $(ls -d $R/output/checkpoint-*/trainer_state.json | sort -V | tail -1))
log "best checkpoint: $BEST"
O=runs/htr/nemotron_parse/test_read; mkdir -p $O
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s + 1)) timeout 3h packages/htr_train/envs/nemotron_parse/.venv/bin/python \
    runs/htr/verify/nemotron_parse_test_read.py "$BEST" $s 2 $O/shard$s.jsonl > $O/shard$s.log 2>&1 &
done
wait
cat $O/shard*.jsonl > $O/pages.jsonl
log "read $(wc -l < $O/pages.jsonl) of 200 test pages"
COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $O/pages.jsonl > $O/cer.txt 2>&1; log "scored: $O/cer.txt"
