#!/bin/bash
# 25 Nov: the pool pages the server's repetition stop cut short, read again on the CPU with the guarded reader
# (runs/htr/curate/prototypes/reread_cpu.py), three shards of 8 threads at low priority; resumable.
cd /home/gabriel/Desktop/diffu2 || exit 1
PY=packages/htr_train/envs/nemotron_parse/.venv/bin/python
for k in 0 1 2; do
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 nice -n 15 $PY runs/htr/curate/prototypes/reread_cpu.py $k 3 8 pool_v3 pool_v2 trusted_v1 > runs/htr/curate/pool_read/reread_cpu.shard$k.log 2>&1 &
done
wait
echo "[$(date '+%F %T')] done"
