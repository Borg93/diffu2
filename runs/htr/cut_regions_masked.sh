#!/bin/bash
# Every GT-Lejonet text region cut as OpenBMB's SDK cuts a layout region (white outside its outline, its lines kept),
# split by test_0050, scaled to at most 9 x 448^2 px, with the lines' dev volumes held out: minicpm_regions_masked.
# Low priority (nice 15, 32 processes) beside the training.
cd /home/gabriel/Desktop/diffu2
D=data/htr
nice -n 15 uv run diffu-reader-set --xml /mnt/work/GT-Lejonet_HTR_202602 --model minicpm-ocr --task elements \
  --max-crop-pixels 1806336 --mask-regions --split-from-gt-test-lists --gt-test-list test_0050 --workers 32 \
  --out $D/minicpm_regions_masked.jsonl && \
uv run htr-train split-dev $D/minicpm_regions_masked.jsonl --like $D/minicpm_lines_scaled.dev.jsonl && echo MASKED REGIONS DONE
