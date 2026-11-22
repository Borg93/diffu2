#!/bin/bash
# 22 Nov: the dev error rate (CER on 115 pages of the 5 dev volumes, books no training set holds) of every checkpoint
# of E60's two runs, as they appear, to RUN/dev_cer.jsonl and to trackio (project diffu, runs RUN-cer).
# GPU 0 until 07:50 on 23 Nov (Gabriel's word for that GPU ends at 08:00), then GPU 2 beside the training.
cd /home/gabriel/Desktop/diffu2 || exit 1
PY=packages/htr_train/envs/nemotron_parse/.venv/bin/python
watch() { CUDA_VISIBLE_DEVICES=$1 $PY -m diffu_htr_train.nemotron_parse.checkpoint_cer "runs/htr/$2" "${@:3}" 2>&1 | grep --line-buffered -v -i 'warn\|deprecat' | sed -u "s/^/[$2] /"; }
if [ "$(date +%s)" -lt "$(date -d '2026-11-23 07:50' +%s)" ]; then
    watch 0 nemotron-parse-selftrain-pretrain --until "2026-11-23 07:50" &
    watch 0 nemotron-parse-selftrain-finetune --until "2026-11-23 07:50" &
    wait
fi
watch 2 nemotron-parse-selftrain-pretrain --idle 30 &
watch 2 nemotron-parse-selftrain-finetune --idle 240 &
wait
