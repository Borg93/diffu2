#!/bin/bash
# Each published model checked as its card says, once it is published: Jev (check_jev.py) on GPU 2 while the draft
# trains on GPU 1, the draft (check_dspark.sh) once the queue is done. Never stops a process: it waits for a free GPU.
# Results in runs/htr/verify/check_jev.log and check_dspark.log; this script's log is watch.log.
cd /home/gabriel/Desktop/diffu2
M=runs/htr/pipeline_markers
V=runs/htr/verify
PY=packages/htr_train/envs/vllm/.venv/bin/python
log() { echo "[$(date '+%F %T')] $*"; }
free() { ! nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | grep -qi "$1"; }  # by PCI bus id
log "waiting for Jev's publish"
until [ -f $M/publish_jev.done ]; do sleep 120; done
until [ -f data/htr/jev_lines.test.jsonl ] && free A3:00; do sleep 60; done
log "checking the published Jev on GPU 2"
CUDA_VISIBLE_DEVICES=2 $PY $V/check_jev.py Gabriel/diffu-minicpm-ocr-htr-jev data/htr/jev_lines.test.jsonl \
  > $V/check_jev.log 2>&1
log "Jev check exit $?: $(tail -3 $V/check_jev.log | tr '\n' ' ')"
log "waiting for the draft's publish"
until [ -f $M/publish_dspark.done ]; do sleep 120; done
until free 75:00 && free A3:00; do sleep 60; done
log "checking the published draft on GPUs 1 and 2"
bash $V/check_dspark.sh > $V/check_dspark.log 2>&1
log "draft check exit $?: $(tail -5 $V/check_dspark.log | tr '\n' ' ')"
