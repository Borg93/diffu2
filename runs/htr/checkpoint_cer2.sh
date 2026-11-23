#!/bin/bash
# 23 Nov: the dev CER of every checkpoint of E60's second fine-tune, on GPU 2 beside it (GPU 0 is not ours).
cd /home/gabriel/Desktop/diffu2 || exit 1
CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/nemotron_parse/.venv/bin/python -m diffu_htr_train.nemotron_parse.checkpoint_cer runs/htr/nemotron-parse-selftrain-finetune2 --idle 600 2>&1 | grep --line-buffered -v -i 'warn\|deprecat'
