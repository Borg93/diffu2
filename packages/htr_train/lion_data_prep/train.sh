#!/bin/bash
# GT-Lejonet's MiniCPM-OCR trained on one server's GPUs (lion_data_prep/configs/minicpm_all.yaml): every line and
# region, on the GPUs $GPUS names, once no process computes on them; then after_minicpm.sh (read only then, so it can
# change while this trains): the trained model's scores and reads, Jev's pairs with those reads, Jev.
# Each step leaves a .done marker in $MARKERS: run it again to resume after the last finished step (a training that
# stopped is resumed by hand from the last checkpoint in its run folder). A file $HOLD holds the trainings back.
#
#   GPUS=1,2 bash packages/htr_train/lion_data_prep/train.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
D=${SETS:-data/htr}
M=${MARKERS:-runs/htr/pipeline_markers}
HOLD=${HOLD:-runs/htr/HOLD}
SPECS=packages/htr_train/lion_data_prep/configs
export HTR_TRAIN_LOCAL_GPUS=${GPUS:-1,2}
mkdir -p "$M"
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step NAME CMD...: run once, stop on failure
  local name=$1; shift
  [ -f "$M/$name.done" ] && { log "skip $name (done)"; return 0; }
  log "start $name"
  if "$@"; then touch "$M/$name.done"; log "done $name"; else log "FAILED $name"; return 1; fi
}
gpus_free() {  # until no process computes on the GPUs $GPUS names (a smoke run left there must not share them)
  local uuids busy
  uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v want=",$HTR_TRAIN_LOCAL_GPUS," \
    'index(want, "," $1 ",") {print $2}')
  while busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -cFf <(echo "$uuids")); \
        [ "$busy" -gt 0 ]; do
    log "GPUs $HTR_TRAIN_LOCAL_GPUS busy ($busy processes): waiting"; sleep 60
  done
}
while [ -f "$HOLD" ]; do log "held: remove $HOLD to train"; sleep 60; done

gpus_free; step minicpm_train uv run htr-train launch $SPECS/minicpm_all.yaml --where local || exit 1
exec bash packages/htr_train/lion_data_prep/after_minicpm.sh
