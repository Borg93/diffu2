#!/bin/bash
# 25 Nov: E72's second GPU window. Pauses the pool read (resumable), then on GPU 2 alone: the guarded and unguarded
# evaluation of the SFT checkpoint and of pilot 1 (dev + test, page by page), pilot 2 (targeted pages, 256 updates),
# the evaluation of pilot 2's checkpoints, and the pool read again with our draft. Log: runs/htr/rlvr_queue2.log
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E2=/mnt/scratch/gabriel/htr/rlvr/eval; R=/mnt/scratch/gabriel/htr/rlvr/runs; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python; mkdir -p $E2
log "pausing the pool read at $(wc -l < runs/htr/curate/pool_read/pool_v3_vllm.jsonl) pages of pool_v3"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 30
ev() { log "eval $2"; runs/htr/rlvr/eval_pilot.sh "$1" "$2" > $E2/$2.log 2>&1 && log "eval $2 done" || log "eval $2 FAILED: $(tail -n 3 $E2/$2.log | cut -c1-300)"; }
ev runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000 sft
ev $R/pilot_1124_2254/checkpoints/step_128/policy/weights/model/consolidated pilot1
$PY runs/htr/rlvr/compare_eval.py sft pilot1 --json $E2/sft_vs_pilot1.json > $E2/sft_vs_pilot1.txt 2>&1; log "compared: $E2/sft_vs_pilot1.txt"
P=pilot2_$(date +%m%d_%H%M); log "pilot 2: $P"
if PROFILE=pilot2 CONFIG=runs/htr/rlvr/grpo_htr_v2.yaml bash runs/htr/rlvr/run_rlvr.sh $P > /dev/null 2>&1; then
  log "pilot 2 ended: $(grep -a 'RL pages\|selected_with_a_failed' $R/$P.log | tail -n 1 | cut -c1-400)"
  for s in 64 128 192 256; do ev $R/$P/checkpoints/step_$s/policy/weights/model/consolidated ${P}_s$s; done
  $PY runs/htr/rlvr/compare_eval.py sft ${P}_s256 --json $E2/sft_vs_${P}.json > $E2/sft_vs_${P}.txt 2>&1; log "compared: $E2/sft_vs_${P}.txt"
else log "pilot 2 FAILED: $(tail -n 5 $R/$P.log | cut -c1-300)"; fi
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
