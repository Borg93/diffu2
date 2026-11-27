#!/bin/bash
# 27 Nov 21:20: GPU 2's driver wedged at ~20:30 (another session's headless Chromium WebGPU processes stuck in D state;
# `nvidia-smi -r` hung too), so the machine was rebooted. This resumes GPU 2's queue in order: SGLang's final stacks
# and the read-back (moss_bench_quant2.sh, NOWAIT), then pilot 4's checkpoint 192 evaluation and dev974 (the two that
# failed at 20:36), then the pool read on pool_v4. Run in tmux: tmux new -d -s gpu2 "bash runs/htr/after_reboot.sh"
# Log: runs/htr/after_reboot.log (and moss_bench_quant.log for the stacks).
cd /home/gabriel/Desktop/diffu2 || exit 1
exec >> runs/htr/after_reboot.log 2>&1
log() { echo "[$(date '+%F %T')] $*"; }
timeout 30 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv || { log "nvidia-smi does not answer: stopping"; exit 1; }
log "SGLang final stacks"
NOWAIT=1 bash runs/htr/moss_bench_quant2.sh >> runs/htr/moss_bench_quant.log 2>&1
log "RL evaluations"
P=pilot4_t03_1127_0841; E2=/mnt/scratch/gabriel/htr/rlvr/eval; R=/mnt/scratch/gabriel/htr/rlvr/runs; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
V=/mnt/scratch/gabriel/envs/parse_rlvr/venv; export CUDA_HOME=$V/lib/python3.13/site-packages/nvidia/cu13; export PATH=$CUDA_HOME/bin:$V/bin:$PATH
export CUDA_VISIBLE_DEVICES=2
runs/htr/rlvr/eval_pilot.sh $R/$P/checkpoints/step_192/policy/weights/model/consolidated ${P}_s192 > $E2/${P}_s192.log 2>&1 && log "eval s192 done" || log "eval s192 FAILED: $(tail -n 3 $E2/${P}_s192.log | cut -c1-300)"
$PY runs/htr/rlvr/compare_eval.py sft ${P}_s192 --json $E2/sft_vs_${P}_s192.json > $E2/sft_vs_${P}_s192.txt 2>&1
DATA=/mnt/scratch/gabriel/htr/rlvr/data_dev_all SPLITS=monitor SKIP=vllm,served runs/htr/rlvr/eval_pilot.sh $R/$P/checkpoints/step_192/policy/weights/model/consolidated ${P}_s192_dev974 > $E2/${P}_s192_dev974.log 2>&1 && log "dev974 done" || log "dev974 FAILED: $(tail -n 3 $E2/${P}_s192_dev974.log | cut -c1-300)"
$PY runs/htr/rlvr/compare_eval.py sft_dev974 ${P}_s192_dev974 --json $E2/dev974_sft_vs_${P}_s192.json > $E2/dev974_sft_vs_${P}_s192.txt 2>&1
$PY runs/htr/rlvr/compare_eval.py pilot2b_s256_dev974 ${P}_s192_dev974 --json $E2/dev974_pilot2b_vs_${P}_s192.json > $E2/dev974_pilot2b_vs_${P}_s192.txt 2>&1
log "pool read (pool_v4)"
tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started"
