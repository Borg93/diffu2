#!/bin/bash
# The two-GPU smoke runs on dmlpai01, one after the other.
cd /home/gabriel/Desktop/diffu2
export HTR_TRAIN_LOCAL_GPUS=1,2
for s in smoke_minicpm_2gpu smoke_jev_2gpu; do
  echo "[$(date '+%F %T')] $s"
  uv run htr-train launch packages/htr_train/configs/dmlpai01/$s.yaml --where local > runs/htr/$s.log 2>&1
  echo "[$(date '+%F %T')] $s exit $?"
done
