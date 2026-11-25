#!/bin/bash
# After the LightOnOCR-3-4B run (13 Nov, the user's order): its best checkpoint reads the 200 test pages Unlimited-OCR 2
# read (both GPUs, 2 h at most, its CER scored), then Nemotron 3.5 Parse's smoke on GPUs 1,2, then, only if the smoke
# ends well, its full run.
cd /home/gabriel/Desktop/diffu2 || exit 1
L=packages/htr_train/lion_data_prep
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q -E "lightonocr3-all (finished|stopped 10 times)" runs/htr/after_uocr2.log; do sleep 120; done
log "lightonocr3-all ended: $(grep -E 'lightonocr3-all (finished|stopped)' runs/htr/after_uocr2.log | tail -1)"
R=runs/htr/lightonocr3-all
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'] or '')" \
  $(ls -d $R/output/checkpoint-*/trainer_state.json | sort -V | tail -1))
log "best checkpoint: $BEST"
O=runs/htr/lightonocr3/test_read; mkdir -p $O
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s + 1)) timeout 2h packages/htr_train/envs/minicpm/.venv/bin/python \
    runs/htr/verify/lightonocr3_test_read.py "$BEST" $s 2 $O/shard$s.jsonl > $O/shard$s.log 2>&1 &
done
wait
cat $O/shard*.jsonl > $O/pages.jsonl
log "read $(wc -l < $O/pages.jsonl) of 200 test pages"
COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $O/pages.jsonl > $O/cer.txt 2>&1; log "scored: $O/cer.txt"
log "Nemotron 3.5 Parse smoke"
if GPUS=1,2 RETRIES=0 bash $L/train_pages.sh $L/configs/smoke_nemotron_parse_2gpu.yaml smoke-nemotron-parse-2gpu; then
  log "smoke ended well: the full run"
  GPUS=1,2 bash $L/train_pages.sh $L/configs/nemotron_parse_all.yaml nemotron-parse-all
else
  log "smoke failed: no full run (see runs/htr/smoke-nemotron-parse-2gpu)"
fi
