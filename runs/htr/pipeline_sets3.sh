#!/bin/bash
# dmlpai01: every GT-Lejonet HTR line and region cut (56 processes) as the GT's TrOCR baseline is trained and tested:
# split by test_0050 (its test pages), lines masked white outside their outline, lines with a transcriber's doubt or
# comment (??, (?), [ ]) or under 10 px left out; regions scaled to at most 9 x 448^2 px (the most MiniCPM-OCR's 9
# slices see). Each with its dev split, then the training (runs/htr/pipeline_train.sh, read only when it starts).
# Each step leaves a .done marker: run again to resume.
set -uo pipefail
cd /home/gabriel/Desktop/diffu2
GT=/mnt/work/GT-Lejonet_HTR_202602
D=data/htr
M=runs/htr/pipeline_markers
SPLIT=(--split-from-gt-test-lists --gt-test-list test_0050)
mkdir -p $M
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step NAME CMD...: run once, stop on failure
  local name=$1; shift
  [ -f $M/$name.done ] && { log "skip $name (done)"; return 0; }
  log "start $name"
  if "$@"; then touch $M/$name.done; log "done $name"; else log "FAILED $name"; return 1; fi
}
step lines uv run diffu-reader-set --xml $GT --model minicpm-ocr --task lines --max-crop-pixels 200704 "${SPLIT[@]}" \
    --mask-lines --reject-uncertain --min-line-side 10 --workers 56 --out $D/minicpm_lines_scaled.jsonl || exit 1
step lines_dev uv run htr-train split-dev $D/minicpm_lines_scaled.jsonl --share 0.02 || exit 1
step regions uv run diffu-reader-set --xml $GT --model minicpm-ocr --task elements --max-crop-pixels 1806336 \
    "${SPLIT[@]}" --workers 56 --out $D/minicpm_regions.jsonl || exit 1
step regions_dev uv run htr-train split-dev $D/minicpm_regions.jsonl --like $D/minicpm_lines_scaled.dev.jsonl || exit 1
log "sets ready: training"
exec bash runs/htr/pipeline_train.sh
