#!/bin/bash
# 26 Nov: when the pool read has finished pool_v3, the temperature test (runs/htr/rlvr/headroom.py: 8 sampled readings
# a page at 0.3, 0.6 and 1.0 against greedy) on the 115 monitor pages, for SFT and for pilot 2b's checkpoint 256; then
# the pool read goes on (pool_v4). GPU 2. Log: runs/htr/rlvr_headroom.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E=/mnt/scratch/gabriel/htr/rlvr/eval; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
until grep -q "pool_v3 read:" runs/htr/pool_rest_read.log; do
  tmux has-session -t =poolread 2>/dev/null || { log "the pool read is not running: going on"; break; }
  sleep 60
done
log "pausing the pool read: pool_v3 $(wc -l < runs/htr/curate/pool_read/pool_v3_vllm.jsonl) pages"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 45
for name in sft_dev974 pilot2b_s256_dev974; do
  log "headroom $name"
  CUDA_VISIBLE_DEVICES=2 HTR_RLVR_CONFIG=$PWD/runs/htr/rlvr/grpo_htr_v2.yaml PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr \
    $PY runs/htr/rlvr/headroom.py $E/$name/view $E/headroom_${name%_dev974} --temperatures 0.3,0.6,1.0 --samples 8 > $E/headroom_${name%_dev974}.log 2>&1 \
    && log "headroom $name done: $(grep -c '^temperature' $E/headroom_${name%_dev974}.log) temperatures" || log "headroom $name FAILED: $(tail -n 3 $E/headroom_${name%_dev974}.log | cut -c1-300)"
done
# RESUME=0: GPU 2 is left free afterwards (26 Nov evening: the user measures the generator's throughput on it).
if [ "${RESUME:-1}" = 1 ]; then
  sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
else
  log "GPU 2 left free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
fi
