#!/bin/bash
# After the Unlimited-OCR 2 run (12 Nov, the user's order): its best checkpoint reads 200 test pages (both GPUs, 2 h at
# most, its CER scored), then LightOnOCR-3-4B's smoke on GPUs 1,2, then, only if the smoke ends well, its full run.
cd /home/gabriel/Desktop/diffu2 || exit 1
L=packages/htr_train/lion_data_prep
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q -E "uocr2-pages (finished|stopped 10 times)" runs/htr/uocr2/train.log; do sleep 120; done
log "uocr2-pages ended: $(grep -E 'uocr2-pages (finished|stopped)' runs/htr/uocr2/train.log | tail -1)"
R=runs/htr/uocr2-pages
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'] or '')" \
  $(ls -d $R/output/checkpoint-*/trainer_state.json | sort -V | tail -1))
log "best checkpoint: $BEST"
mkdir -p runs/htr/uocr2/test_read
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$((s + 1)) timeout 2h packages/htr_train/envs/uocr2/.venv/bin/python runs/htr/verify/uocr2_test_read.py \
    "$BEST" $s 2 runs/htr/uocr2/test_read/shard$s.jsonl > runs/htr/uocr2/test_read/shard$s.log 2>&1 &
done
wait
cat runs/htr/uocr2/test_read/shard*.jsonl > runs/htr/uocr2/test_read/pages.jsonl
log "read $(wc -l < runs/htr/uocr2/test_read/pages.jsonl) of 200 test pages"
COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer runs/htr/uocr2/test_read/pages.jsonl \
  > runs/htr/uocr2/test_read/cer.txt 2>&1; log "scored: runs/htr/uocr2/test_read/cer.txt"
log "LightOnOCR-3-4B smoke"
if GPUS=1,2 RETRIES=0 bash $L/train_pages.sh $L/configs/smoke_lightonocr3_2gpu.yaml smoke-lightonocr3-2gpu; then
  log "smoke ended well: the full run"
  GPUS=1,2 bash $L/train_pages.sh $L/configs/lightonocr3_all.yaml lightonocr3-all
else
  log "smoke failed: no full run (see runs/htr/smoke-lightonocr3-2gpu)"
fi
