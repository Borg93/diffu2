#!/bin/bash
# CURATION E43: the old pipeline (YOLO regions and lines, then TrOCR, as htrflow 0.2.6 runs them) reads the 200 test
# pages and is scored on the whole page as the page readers are. htrflow lives in an environment of its own under
# /mnt/scratch/gabriel/uv-cache-htrflow (not added to the project); the models go to /mnt/scratch/gabriel/models/hf_hub.
#   GPU=2 bash runs/htr/old_pipeline_test.sh [OUT [LIMIT]]      (GPU 1 or 2 only)
cd /home/gabriel/Desktop/diffu2 || exit 1
GPU=${GPU:-2}; case "$GPU" in 1|2) ;; *) echo "GPU must be 1 or 2"; exit 1;; esac
O=${1:-runs/htr/test_read_old_pipeline}; S=/mnt/scratch/gabriel
export UV_CACHE_DIR=$S/uv-cache-htrflow HF_HUB_CACHE=$S/models/hf_hub HF_HUB_DISABLE_IMPLICIT_TOKEN=1 YOLO_CONFIG_DIR=$S/uv-cache-htrflow/ultralytics
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
# Each recogniser needs its own transformers: 5.23, which htrflow 0.2.6 resolves to today, cannot run the 2024 one's
# decoder ("Cannot copy out of meta tensor"), and 4.57 cannot read the tokenizer file of the 2026 one, saved by 5.
OLD=(uvx --from htrflow==0.2.6 --with 'transformers<5'); NEW=(uvx --from htrflow==0.2.6)
log() { echo "[$(date '+%F %T')] $*"; }
uv run python runs/htr/verify/old_pipeline_test_read.py prepare $O $2 || exit 1
"${NEW[@]}" python -c "from ultralytics import settings; settings.update({'sync': False}); print('ultralytics sync:', settings['sync'])" 2>&1 | tail -n 1
for run in ${RUNS:-swe2 lion}; do
  if [ $run = swe2 ]; then HTRFLOW=("${OLD[@]}"); else HTRFLOW=("${NEW[@]}"); fi
  log "$run: reading"
  "${HTRFLOW[@]}" htrflow pipeline $O/$run.yaml $O/images --logfile $O/$run.log --batch-output 25 || { log "FAILED $run"; continue; }
  for order in lines_order reading_order; do
    uv run python runs/htr/verify/old_pipeline_test_read.py score $O $run $order
  done
  log "$run: read"
done
for run in ${RUNS:-swe2 lion}; do  # htr-eval's tables last: its edit distance is plain Python, minutes a file
  for order in lines_order reading_order; do
    [ -s $O/${run}_$order.jsonl ] || continue
    COLUMNS=200 nice -n 10 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $O/${run}_$order.jsonl > $O/${run}_$order.cer.txt 2>&1
    log "$run $order: $(grep -m 1 '│ all' $O/${run}_$order.cer.txt | cut -c1-200)"
  done
done
log "done"
