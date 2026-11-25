#!/bin/bash
# 25 Nov: GPU 2 back after being lent. Evaluates pilot 2b's checkpoint 64 (four readers, dev and test), compares it
# with SFT and pilot 1, then starts the pool read again (resumable). Log: runs/htr/resume_after_lend.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E2=/mnt/scratch/gabriel/htr/rlvr/eval; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
C=/mnt/scratch/gabriel/htr/rlvr/runs/pilot2b_1125_1014/checkpoints/step_64/policy/weights/model/consolidated
log "eval pilot2b_s64"
runs/htr/rlvr/eval_pilot.sh $C pilot2b_s64 > $E2/pilot2b_s64.log 2>&1 && log "eval done" || log "eval FAILED: $(tail -n 3 $E2/pilot2b_s64.log | cut -c1-300)"
$PY runs/htr/rlvr/compare_eval.py sft pilot2b_s64 --json $E2/sft_vs_pilot2b_s64.json > $E2/sft_vs_pilot2b_s64.txt 2>&1
$PY runs/htr/rlvr/compare_eval.py pilot1 pilot2b_s64 --json $E2/pilot1_vs_pilot2b_s64.json > $E2/pilot1_vs_pilot2b_s64.txt 2>&1
log "compared: $E2/sft_vs_pilot2b_s64.txt"
tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
