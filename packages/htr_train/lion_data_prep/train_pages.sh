#!/bin/bash
# A page reader trained on one server's GPUs: SPEC launched as run NAME (htr-train launch --where local), on the GPUs
# $GPUS names once no process computes on them. A training that stops (GPU 2 hangs now and then with Xid 8) is started
# again up to $RETRIES times (10), 2 minutes apart, from its own command.sh: htr-train's steps resume from the newest
# whole checkpoint. Run it in tmux, so it survives a disconnect.
#
#   GPUS=1,2 bash packages/htr_train/lion_data_prep/train_pages.sh packages/htr_train/lion_data_prep/configs/uocr2_pages.yaml uocr2-pages
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
SPEC=$1
NAME=$2
RUN=${HTR_TRAIN_RUNS:-runs/htr}/$NAME
RETRIES=${RETRIES:-10}
export HTR_TRAIN_LOCAL_GPUS=${GPUS:-1,2}
log() { echo "[$(date '+%F %T')] $*"; }
gpus_free() {  # until no process computes on the GPUs $GPUS names
  local uuids busy
  uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v want=",$HTR_TRAIN_LOCAL_GPUS," \
    'index(want, "," $1 ",") {print $2}')
  while busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -cFf <(echo "$uuids")); \
        [ "$busy" -gt 0 ]; do
    log "GPUs $HTR_TRAIN_LOCAL_GPUS busy ($busy processes): waiting"; sleep 60
  done
}
gpus_free
if [ ! -f "$RUN/manifest.json" ]; then
  log "launching $NAME from $SPEC on GPUs $HTR_TRAIN_LOCAL_GPUS"
  uv run htr-train launch "$SPEC" --where local && { log "$NAME finished"; exit 0; }
  [ -f "$RUN/manifest.json" ] || { log "$NAME refused before it started"; exit 1; }
else
  log "resuming $NAME from its command.sh"
  bash "$RUN/command.sh" && { log "$NAME finished"; exit 0; }
fi
for attempt in $(seq "$RETRIES"); do
  log "$NAME stopped: starting it again ($attempt of $RETRIES) in 2 minutes, from its newest checkpoint"
  sleep 120; gpus_free
  bash "$RUN/command.sh" && { log "$NAME finished"; exit 0; }
done
log "$NAME stopped $RETRIES times more: giving up"
exit 1
