#!/bin/bash
# 24 Nov: the reader reads the pool pages not read yet (pool_v3, v4, v5: 3.24 million), through vLLM on GPU 2, with the
# reader E60 gave (nemotron-parse-selftrain-finetune2 checkpoint-3000: test CER 6.63 %, dev 4.96 %; pool_v2 and the
# trusted pages were read by nemotron-parse-more, 7.55 %). Resumable: stop with tmux kill-session -t poolread, start again.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
tmux new-window -d -t htr -n parse2 "GPU=2 PORT=8012 FRACTION=0.85 SEQS=${SEQS:-128} DRAFT=${DRAFT:-16} bash runs/htr/parse_serve.sh ${SERVE:-$S/serve/parse-selftrain2} 2>&1 | tee runs/htr/verify/parse_serve_selftrain2.log"
until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parse2 || { log "FAILED server"; exit 1; }; done
for v in 3 4 5; do
  log "server up: reading pool_v$v"
  for try in 1 2 3; do
    SET=$S/pool_v${v}_nemotron/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $O/pool_v${v}_vllm.jsonl ${SEQS:-128} 2>&1 | tee -a $O/pool_v${v}_vllm.log | tail -n 1
  done
  log "pool_v$v read: $(wc -l < $O/pool_v${v}_vllm.jsonl)"
done
tmux kill-window -t htr:parse2
