#!/bin/bash
# Unlimited-OCR 2's page set (11 Nov, the user's choice of whole-page reader): every GT page, an opening cut into its
# two pages at its gutter (an opening its lines cannot split left out), answered with its blocks in the transcribers'
# reading order, each with its label and box (diffu-reader-set --model unlimited-ocr-2 --task parse); test pages by
# test_0050, as every other set here; then the dev split by the same volumes as the MiniCPM-OCR regions' dev split.
cd /home/gabriel/Desktop/diffu2 || exit 1
GT=/mnt/work/GT-Lejonet_HTR_202602
D=data/htr
log() { echo "[$(date '+%F %T')] $*"; }
log "pages"
uv run diffu-reader-set --xml "$GT" --model unlimited-ocr-2 --task parse \
  --split-from-gt-test-lists --gt-test-list test_0050 --workers 56 --out "$D/uocr2_pages.jsonl" || { log "FAILED pages"; exit 1; }
log "done pages: $(wc -l < $D/uocr2_pages.jsonl) training, $(wc -l < $D/uocr2_pages.test.jsonl) test"
