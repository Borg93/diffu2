#!/bin/bash
# 26 Nov: SFT and pilot 2b's checkpoint 256 read by the Transformers guard (the reader in use) on all 974 dev pages
# (5 volumes no training set holds), compared page by page; then the pool read again. GPU 2. Log: runs/htr/rlvr_dev974.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E2=/mnt/scratch/gabriel/htr/rlvr/eval; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
log "pausing the pool read at $(wc -l < runs/htr/curate/pool_read/pool_v3_vllm.jsonl) pages of pool_v3"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 30
for pair in "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000 sft_dev974" \
            "/mnt/scratch/gabriel/htr/rlvr/runs/pilot2b_1125_1641/checkpoints/step_256/policy/weights/model/consolidated pilot2b_s256_dev974"; do
  set -- $pair; log "eval $2"
  DATA=/mnt/scratch/gabriel/htr/rlvr/data_dev_all SPLITS=monitor SKIP=vllm,served runs/htr/rlvr/eval_pilot.sh "$1" "$2" > $E2/$2.log 2>&1 && log "eval $2 done" || log "eval $2 FAILED: $(tail -n 3 $E2/$2.log | cut -c1-300)"
done
$PY runs/htr/rlvr/compare_eval.py sft_dev974 pilot2b_s256_dev974 --json $E2/dev974_sft_vs_pilot2b_s256.json > $E2/dev974_sft_vs_pilot2b_s256.txt 2>&1; log "compared: $E2/dev974_sft_vs_pilot2b_s256.txt"
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
