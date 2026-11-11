#!/bin/bash
# GT-Lejonet's page set for a whole-page reader (default Unlimited-OCR 2, --task parse): every GT page, an opening cut
# into its two pages at its gutter (an opening its lines cannot split left out, as every spread), answered with its
# blocks in the transcribers' reading order, each with its label and box; test pages by test_0050, as every other set
# here; then the dev split by the volumes of the MiniCPM-OCR sets' (htr-train split-dev --like), so dev scores compare.
#
#   GT=/mnt/work/GT-Lejonet_HTR_202602 bash packages/htr_train/lion_data_prep/build_pages.sh
#   MODEL=... TASK=... OUT=... for another reader's format
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GT=${GT:-$HOME/Desktop/data/GT-Lejonet_HTR_202602}
D=${SETS:-data/htr}
MODEL=${MODEL:-unlimited-ocr-2}
TASK=${TASK:-parse}
OUT=${OUT:-$D/uocr2_pages.jsonl}
WORKERS=${WORKERS:-$(nproc)}
log() { echo "[$(date '+%F %T')] $*"; }
log "pages ($MODEL, $TASK) -> $OUT"
uv run diffu-reader-set --xml "$GT" --model "$MODEL" --task "$TASK" \
  --split-from-gt-test-lists --gt-test-list test_0050 --workers "$WORKERS" --out "$OUT" || { log "FAILED pages"; exit 1; }
uv run htr-train split-dev "$OUT" --like "$D/minicpm_lines_scaled.dev.jsonl" || { log "FAILED dev split"; exit 1; }
log "done: $(wc -l < "${OUT%.jsonl}.train.jsonl") training, $(wc -l < "${OUT%.jsonl}.dev.jsonl") dev, $(wc -l < "${OUT%.jsonl}.test.jsonl") test pages"
