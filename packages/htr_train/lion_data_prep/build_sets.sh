#!/bin/bash
# GT-Lejonet's training sets for MiniCPM-OCR, cut as the GT's TrOCR baseline is trained and tested (README.md): every
# HTR line (split by test_0050, masked white outside its outline, a line with a transcriber's doubt or comment or under
# 10 px left out, scaled to one 448 x 448 view) and every region (scaled to at most 9 x 448^2 px, the most
# MiniCPM-OCR's 9 slices see), each with a dev split of whole volumes; then train.sh, unless NO_TRAIN is set.
# Each step leaves a .done marker in $MARKERS: run it again to resume after the last finished step.
#
#   GT=/mnt/work/GT-Lejonet_HTR_202602 WORKERS=56 bash packages/htr_train/lion_data_prep/build_sets.sh
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GT=${GT:-$HOME/Desktop/data/GT-Lejonet_HTR_202602}
D=${SETS:-data/htr}
WORKERS=${WORKERS:-$(nproc)}
M=${MARKERS:-runs/htr/pipeline_markers}
SPLIT=(--split-from-gt-test-lists --gt-test-list test_0050)
mkdir -p "$M"
log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step NAME CMD...: run once, stop on failure
  local name=$1; shift
  [ -f "$M/$name.done" ] && { log "skip $name (done)"; return 0; }
  log "start $name"
  if "$@"; then touch "$M/$name.done"; log "done $name"; else log "FAILED $name"; return 1; fi
}
step lines uv run diffu-reader-set --xml "$GT" --model minicpm-ocr --task lines --max-crop-pixels 200704 "${SPLIT[@]}" \
    --mask-lines --reject-uncertain --min-line-side 10 --workers "$WORKERS" --out "$D/minicpm_lines_scaled.jsonl" || exit 1
step lines_dev uv run htr-train split-dev "$D/minicpm_lines_scaled.jsonl" --share 0.02 || exit 1
step regions uv run diffu-reader-set --xml "$GT" --model minicpm-ocr --task elements --max-crop-pixels 1806336 \
    --mask-regions "${SPLIT[@]}" --workers "$WORKERS" --out "$D/minicpm_regions_masked.jsonl" || exit 1
step regions_dev uv run htr-train split-dev "$D/minicpm_regions_masked.jsonl" \
    --like "$D/minicpm_lines_scaled.dev.jsonl" || exit 1
# the out-of-domain lines, and the fixed draws the training's cer callback reads at each evaluation
E=packages/htr_train/envs/minicpm/.venv/bin/htr-eval
C=$D/eval/in_training
mkdir -p "$C"
step ood_set $E ood-set --out "$D/eval/ood.jsonl" || exit 1
step cer_sets bash -c "$E set-lines $D/minicpm_lines_scaled.dev.jsonl --sample 1000 --seed 0 --out $C/dev_lines.jsonl \
  && $E set-lines $D/minicpm_regions_masked.dev.jsonl --sample 200 --seed 0 --out $C/dev_regions.jsonl \
  && $E draw $D/eval/ood.jsonl --count 500 --seed 0 --out $C/ood.jsonl" || exit 1
log "sets ready"
[ -n "${NO_TRAIN:-}" ] || exec bash packages/htr_train/lion_data_prep/train.sh
