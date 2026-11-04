#!/bin/bash
# GT-Lejonet's readers trained on one server's GPUs (lion_data_prep/configs): MiniCPM-OCR on every line and
# region, Jev's pairs built meanwhile on the CPU, then Jev; both on the GPUs $GPUS names, each waiting until no
# process computes on them.
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

(   # Jev's pairs from the training lines and its dev split; test pairs to score it on
  step jev_pairs uv run htr-train jev-set --lines "$D/minicpm_lines_scaled.train.jsonl" --out "$D/jev_lines.jsonl" &&
  step jev_dev uv run htr-train split-dev "$D/jev_lines.jsonl" --share 0.02 &&
  step jev_test_pairs uv run htr-train jev-set --side test --lines "$D/minicpm_lines_scaled.test.jsonl" \
      --out "$D/jev_lines.test.jsonl"
) > runs/htr/pipeline_jev_pairs.log 2>&1 &
PAIRS=$!

gpus_free; step minicpm_train uv run htr-train launch $SPECS/minicpm_all.yaml --where local || exit 1
wait $PAIRS || { log "FAILED Jev's pairs (runs/htr/pipeline_jev_pairs.log)"; exit 1; }
gpus_free; step jev_train uv run htr-train launch $SPECS/jev_all.yaml --where local || exit 1
log "training finished"
