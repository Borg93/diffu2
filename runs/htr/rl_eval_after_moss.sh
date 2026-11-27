#!/bin/bash
# 27 Nov 13:45: pilot 4 was cut at update 241 by my closing its tmux session (its last checkpoint is 192), so its
# evaluation is of checkpoints 64, 128 and 192 (192 on all 974 dev pages); it runs after the generator tests now on
# GPU 2 after the generator tests (moss_bench_set40.sh, run from here: 13:40 my stopping a parent shell killed the first
# run of it as well), then the pool read. Log: runs/htr/rl_eval_after_moss.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
P=pilot4_t03_1127_0841; E2=/mnt/scratch/gabriel/htr/rlvr/eval; R=/mnt/scratch/gabriel/htr/rlvr/runs; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
WAIT=0 bash runs/htr/moss_bench_set40.sh >> runs/htr/moss_bench_set40.log 2>&1  # the generator tests first, then the RL evaluations
log "generator tests ended: RL evaluations"
V=/mnt/scratch/gabriel/envs/parse_rlvr/venv; export CUDA_HOME=$V/lib/python3.13/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$V/bin:$PATH
ev() { log "eval $2"; runs/htr/rlvr/eval_pilot.sh "$1" "$2" > $E2/$2.log 2>&1 && log "eval $2 done" || log "eval $2 FAILED: $(tail -n 3 $E2/$2.log | cut -c1-300)"; }
for s in 64 128 192; do ev $R/$P/checkpoints/step_$s/policy/weights/model/consolidated ${P}_s$s; done
for s in 64 128 192; do $PY runs/htr/rlvr/compare_eval.py sft ${P}_s$s --json $E2/sft_vs_${P}_s$s.json > $E2/sft_vs_${P}_s$s.txt 2>&1; done
log "eval ${P}_s192 on all 974 dev pages"
DATA=/mnt/scratch/gabriel/htr/rlvr/data_dev_all SPLITS=monitor SKIP=vllm,served runs/htr/rlvr/eval_pilot.sh $R/$P/checkpoints/step_192/policy/weights/model/consolidated ${P}_s192_dev974 > $E2/${P}_s192_dev974.log 2>&1 || log "dev974 FAILED: $(tail -n 3 $E2/${P}_s192_dev974.log | cut -c1-300)"
$PY runs/htr/rlvr/compare_eval.py sft_dev974 ${P}_s192_dev974 --json $E2/dev974_sft_vs_${P}_s192.json > $E2/dev974_sft_vs_${P}_s192.txt 2>&1
$PY runs/htr/rlvr/compare_eval.py pilot2b_s256_dev974 ${P}_s192_dev974 --json $E2/dev974_pilot2b_vs_${P}_s192.json > $E2/dev974_pilot2b_vs_${P}_s192.txt 2>&1
log "compared: $E2/sft_vs_${P}_s*.txt, $E2/dev974_*_${P}_s192.txt"
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started (pool_v4)"
