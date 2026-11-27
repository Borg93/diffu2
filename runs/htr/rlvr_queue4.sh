#!/bin/bash
# 27 Nov: pilot 4 = pilot 2b at rollout temperature 0.3 (grpo_htr_v2_t03.yaml), on GPU 2 alone, once the generator's
# torchao runs have ended (tmux mossbench gone). Then its checkpoints at 64, 128, 192, 256 under the four readers on
# the monitor and test pages, checkpoint 256 on all 974 dev pages (the Transformers guard) against SFT, and the pool
# read on pool_v4. Log: runs/htr/rlvr_queue4.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E2=/mnt/scratch/gabriel/htr/rlvr/eval; R=/mnt/scratch/gabriel/htr/rlvr/runs; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
V=/mnt/scratch/gabriel/envs/parse_rlvr/venv; export CUDA_HOME=$V/lib/python3.13/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$V/bin:$PATH
while tmux has-session -t =mossbench 2>/dev/null; do sleep 60; done
sleep 30; log "GPU 2: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 2)"
ev() { log "eval $2"; runs/htr/rlvr/eval_pilot.sh "$1" "$2" > $E2/$2.log 2>&1 && log "eval $2 done" || log "eval $2 FAILED: $(tail -n 3 $E2/$2.log | cut -c1-300)"; }
P=pilot4_t03_$(date +%m%d_%H%M); log "pilot 4: $P"
if PROFILE=pilot2 CONFIG=runs/htr/rlvr/grpo_htr_v2_t03.yaml bash runs/htr/rlvr/run_rlvr.sh $P > /dev/null 2>&1; then
  log "pilot 4 ended"
  for s in 64 128 192 256; do ev $R/$P/checkpoints/step_$s/policy/weights/model/consolidated ${P}_s$s; done
  for s in 64 128 192 256; do $PY runs/htr/rlvr/compare_eval.py sft ${P}_s$s --json $E2/sft_vs_${P}_s$s.json > $E2/sft_vs_${P}_s$s.txt 2>&1; done
  log "eval ${P}_s256 on all 974 dev pages"
  DATA=/mnt/scratch/gabriel/htr/rlvr/data_dev_all SPLITS=monitor SKIP=vllm,served runs/htr/rlvr/eval_pilot.sh $R/$P/checkpoints/step_256/policy/weights/model/consolidated ${P}_s256_dev974 > $E2/${P}_s256_dev974.log 2>&1 || log "dev974 FAILED: $(tail -n 3 $E2/${P}_s256_dev974.log | cut -c1-300)"
  $PY runs/htr/rlvr/compare_eval.py sft_dev974 ${P}_s256_dev974 --json $E2/dev974_sft_vs_${P}_s256.json > $E2/dev974_sft_vs_${P}_s256.txt 2>&1
  $PY runs/htr/rlvr/compare_eval.py pilot2b_s256_dev974 ${P}_s256_dev974 --json $E2/dev974_pilot2b_vs_${P}_s256.json > $E2/dev974_pilot2b_vs_${P}_s256.txt 2>&1
  log "compared: $E2/sft_vs_${P}_s*.txt, $E2/dev974_*_${P}_s256.txt"
else log "pilot 4 FAILED: $(tail -n 5 $R/$P.log | cut -c1-300)"; fi
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started (pool_v4)"
