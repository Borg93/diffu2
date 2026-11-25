#!/bin/bash
# 25 Nov: GPU 2 is lent to another person for the rest of the day. Waits until pilot 2b has saved its checkpoint at
# update STEP (64 by default) and gone on to the next update, then stops, in this order: the queue (so it does not
# start the pool read again), the RL run, the Ray processes it leaves; and checks that none of ours is left on GPU 2.
# Nothing is started afterwards. Log: runs/htr/rlvr_stop.log
cd /home/gabriel/Desktop/diffu2 || exit 1
STEP=${STEP:-64}
RUN=$(ls -d /mnt/scratch/gabriel/htr/rlvr/runs/pilot2b_*/ | tail -n 1)
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for checkpoint $STEP of $RUN"
until python3 - "$RUN" "$STEP" <<'P'
import json, sys
run, step = sys.argv[1], int(sys.argv[2])
try:
    steps = [json.loads(l)["step"] for l in open(run + "history.jsonl") if '"phase": "train"' in l]
except OSError:
    sys.exit(1)
sys.exit(0 if steps and max(steps) > step else 1)
P
do
  if ! pgrep -u "$USER" -f "stage2_rlvr/train.py" > /dev/null; then log "the RL run is no longer running: stopping only the queue"; break; fi
  sleep 30
done
ls -d "$RUN"checkpoints/step_"$STEP"/policy/weights/model/consolidated > /dev/null 2>&1 && log "checkpoint $STEP saved: $(du -sh "$RUN"checkpoints/step_"$STEP" | cut -f1)" || log "NO checkpoint $STEP found"
for pid in $(pgrep -u "$USER" -f "bash runs/htr/rlvr_queue3.sh"); do kill -KILL "$pid" && log "queue stopped (pid $pid)"; done
for pid in $(pgrep -u "$USER" -f "stage2_rlvr/train.py"); do kill -TERM "$pid" && log "RL run stopped (pid $pid)"; done
for _ in $(seq 18); do pgrep -u "$USER" -f "stage2_rlvr/train.py" > /dev/null || break; sleep 5; done
sleep 10
for pid in $(pgrep -u "$USER" -f "ray::|ray/core/src/ray/(raylet|gcs)"); do kill -KILL "$pid" 2>/dev/null && log "left-over Ray process stopped (pid $pid)"; done
sleep 5
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r p; do [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && echo "$p"; done)
[ -z "$left" ] && log "GPU 2 holds nothing of ours" || log "STILL ON A GPU: $left"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | while read -r line; do log "GPU $line"; done
