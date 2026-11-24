#!/bin/bash
# 24 Nov: E72's first runs on GPU 2, by themselves. Waits for the draft's training (E52) to leave the GPU, pauses the
# pool read (resumable), runs the 2-update smoke; if it passes, the 128-update pilot; then the pool read goes on.
# If the smoke fails the pool read goes on at once. Log: runs/htr/rlvr_queue.log; runs in /mnt/scratch/gabriel/htr/rlvr/runs/
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
resume() { tmux new-session -d -s poolread "SEQS=128 DRAFT=8 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"; log "pool read started again"; }
while pgrep -u "$USER" -f "runs/htr/draft/" > /dev/null; do sleep 30; done
log "the draft's jobs have left GPU 2: pausing the pool read at $(wc -l < runs/htr/curate/pool_read/pool_v3_vllm.jsonl) pages of pool_v3"
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 30
S=smoke_$(date +%m%d_%H%M); P=pilot_$(date +%m%d_%H%M)
log "smoke $S"
if PROFILE=tiny bash runs/htr/rlvr/run_rlvr.sh $S > /dev/null 2>&1 && [ -s /mnt/scratch/gabriel/htr/rlvr/runs/$S/artifact.json ]; then
  log "smoke passed: $(python3 -c "import json;e=json.load(open('/mnt/scratch/gabriel/htr/rlvr/runs/$S/evaluation.json'));print({k:{m:round(v,4) for m,v in e[k].items() if isinstance(v,(int,float))} for k in e})" 2>&1 | cut -c1-600)"
  log "pilot $P"
  if bash runs/htr/rlvr/run_rlvr.sh $P > /dev/null 2>&1; then log "pilot ended: /mnt/scratch/gabriel/htr/rlvr/runs/$P"; else log "pilot FAILED: tail of its log: $(tail -n 5 /mnt/scratch/gabriel/htr/rlvr/runs/$P.log | cut -c1-300)"; fi
else
  log "smoke FAILED: $(tail -n 8 /mnt/scratch/gabriel/htr/rlvr/runs/$S.log | cut -c1-300)"
fi
sleep 30; resume
