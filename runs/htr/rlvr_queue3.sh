#!/bin/bash
# 25 Nov: pilot 2 again (pilot 2b) with the loop rule mended (a loop must also make the answer 1.15 times the truth's
# length: htr_reward.py, grpo_htr_v2.yaml), on GPU 2 alone. Pauses the pool read, runs pilot 2b, evaluates its
# checkpoints at 64, 128, 192, 256 under the four readers, compares with SFT and pilot 1, then the pool read again.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
E2=/mnt/scratch/gabriel/htr/rlvr/eval; R=/mnt/scratch/gabriel/htr/rlvr/runs; PY=/mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python
log "pausing the pool read at $(wc -l < runs/htr/curate/pool_read/pool_v3_vllm.jsonl) pages of pool_v3"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 30
ev() { log "eval $2"; runs/htr/rlvr/eval_pilot.sh "$1" "$2" > $E2/$2.log 2>&1 && log "eval $2 done" || log "eval $2 FAILED: $(tail -n 3 $E2/$2.log | cut -c1-300)"; }
P=pilot2b_$(date +%m%d_%H%M); log "pilot 2b: $P"
if PROFILE=pilot2 CONFIG=runs/htr/rlvr/grpo_htr_v2.yaml bash runs/htr/rlvr/run_rlvr.sh $P > /dev/null 2>&1; then
  log "pilot 2b ended"
  for s in 64 128 192 256; do ev $R/$P/checkpoints/step_$s/policy/weights/model/consolidated ${P}_s$s; done
  for s in 64 128 192 256; do $PY runs/htr/rlvr/compare_eval.py sft ${P}_s$s --json $E2/sft_vs_${P}_s$s.json > $E2/sft_vs_${P}_s$s.txt 2>&1; done
  $PY runs/htr/rlvr/compare_eval.py pilot1 ${P}_s256 --json $E2/pilot1_vs_${P}_s256.json > $E2/pilot1_vs_${P}_s256.txt 2>&1
  log "compared: $E2/sft_vs_${P}_s*.txt, $E2/pilot1_vs_${P}_s256.txt"
else log "pilot 2b FAILED: $(tail -n 5 $R/$P.log | cut -c1-300)"; fi
sleep 30; tmux new-session -d -s poolread "SERVE=/mnt/scratch/gabriel/htr/serve/parse-selftrain2-draftv1 SEQS=128 DRAFT=16 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"
