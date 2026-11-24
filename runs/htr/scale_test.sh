#!/bin/bash
# 24 Nov: the 200 test pages read at smaller image sizes through the pool read's server (CER does not depend on load),
# to see what a cheaper image stage costs in accuracy. Results: runs/htr/nemotron_parse/test_read_selftrain2_scale*/cer.txt
cd /home/gabriel/Desktop/diffu2 || exit 1
for sc in 1 0.85 0.7 0.6 0.5; do
  T=runs/htr/nemotron_parse/test_read_selftrain2_scale$sc; mkdir -p $T
  SCALE=$sc URL=http://127.0.0.1:8012/v1 uv run --with httpx --with pillow python runs/htr/verify/parse_vllm_read.py $T/pages.jsonl 16 2>&1 | tail -n 1 | cut -c1-200
  COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $T/pages.jsonl > $T/cer.txt 2>&1
  echo "scale $sc: $(grep 'all' $T/cer.txt | cut -c1-120)"
done
echo finished
