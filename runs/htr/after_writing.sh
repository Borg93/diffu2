#!/bin/bash
# 21 Nov (Gabriel: yes): GPU 2 after the writing layer over every scan (tmux afterread2) and Jev's trial (jev).
#  1. The 815 pages called turned, turned back (turned_reread.py), read again by our reader through vLLM (CURATION E66).
#  2. E60: E38's pretraining and fine-tune with our reader's readings as the labels (selftrain_set.py), on GPU 2 alone,
#     then the 200 test pages read and scored against E38's 6.92 % (the same recipe with the old labels).
# Stop it with tmux kill-window -t htr:afterwriting (the training resumes from its newest checkpoint when started again).
cd /home/gabriel/Desktop/diffu2 || exit 1
L=packages/htr_train/lion_data_prep; S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'afterread2|jev'; do sleep 60; done
log "the writing layer and Jev's trial have ended"
if [ ! -s $O/turned_v1_vllm.jsonl ] || [ "$(wc -l < $O/turned_v1_vllm.jsonl)" -lt "$(wc -l < $S/turned_v1/pages.jsonl)" ]; then
  tmux new-window -d -t htr -n parse2 "GPU=2 PORT=8012 FRACTION=0.85 SEQS=128 bash runs/htr/parse_serve.sh 2>&1 | tee runs/htr/verify/parse_serve_turned.log"
  until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parse2 || { log "FAILED server"; exit 1; }; done
  SET=$S/turned_v1/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $O/turned_v1_vllm.jsonl 64 2>&1 | tail -n 1
  tmux kill-window -t htr:parse2; sleep 20
fi
log "turned pages read again: $(wc -l < $O/turned_v1_vllm.jsonl)"
uv run python $P/selftrain_set.py $S/trusted_v1_selftrain/pages 2>&1 | grep -v -i warn | tail -n 1 || { log "FAILED set"; exit 1; }
GPUS=2 bash $L/train_pages.sh $L/configs/nemotron_parse_selftrain_pretrain.yaml nemotron-parse-selftrain-pretrain || { log "pretraining failed: no fine-tune"; exit 1; }
GPUS=2 bash $L/train_pages.sh $L/configs/nemotron_parse_selftrain_finetune.yaml nemotron-parse-selftrain-finetune || { log "fine-tune failed"; exit 1; }
R=runs/htr/nemotron-parse-selftrain-finetune; T=runs/htr/nemotron_parse/test_read_selftrain_norepeat10; mkdir -p $T
BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'] or '')" $(ls -d $R/output/checkpoint-*/trainer_state.json | sort -V | tail -1))
log "reading the test pages with $BEST"
NO_REPEAT_NGRAM=10 CUDA_VISIBLE_DEVICES=2 timeout 3h packages/htr_train/envs/nemotron_parse/.venv/bin/python runs/htr/verify/nemotron_parse_test_read.py "$BEST" 0 1 $T/pages.jsonl > $T/read.log 2>&1
log "read $(wc -l < $T/pages.jsonl) of 200 test pages"
COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $T/pages.jsonl > $T/cer.txt 2>&1
uv run --with rapidfuzz python runs/htr/curate/prototypes/compare_reads.py selftrain=test_read_selftrain_norepeat10 oldlabels=test_read_trusted_norepeat10 none=test_read_norepeat10 > $T/compare.txt 2>&1
log "scored: $T/cer.txt, $T/compare.txt"
