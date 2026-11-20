#!/bin/bash
# 20 Nov: waits for the file server's disks, checks what the read needs, then resumes the reading of pool_v2 through
# vLLM on GPU 2 alone (GPU 1 is another person's from 20 Nov). Resumable; stop with tmux kill-window on parse2, resume.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
until timeout 10 mountpoint -q /mnt/scratch && timeout 10 mountpoint -q /mnt/work; do sleep 60; done
log "both disks mounted"
for f in $S/pool_v2_nemotron/pages.jsonl $S/pool_v2_nemotron/pages.meta.jsonl $S/serve/parse-more/config.json $S/serve/parse-more/dflash2/model.safetensors runs/htr/nemotron-parse-more/output/checkpoint-3094/model.safetensors; do
  timeout 60 test -s "$f" || { log "MISSING or unreadable: $f: not starting"; exit 1; }
done
n=$(timeout 600 wc -l < $S/pool_v2_nemotron/pages.jsonl); [ "$n" = 999808 ] || { log "pages.jsonl has $n lines, not 999808: not starting"; exit 1; }
img=$(tail -n 1 $S/pool_v2_nemotron/pages.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['images'][0])"); timeout 60 test -s "$img" || { log "a page image does not read: $img"; exit 1; }
log "the set, the model and the draft read: starting the server on GPU 2"
tmux new-window -d -t htr -n parse2 "GPU=2 PORT=8012 FRACTION=0.85 SEQS=128 bash runs/htr/parse_serve.sh 2>&1 | tee runs/htr/verify/parse_serve_2.log"
until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parse2 || { log "FAILED server"; exit 1; }; done
log "server up: reading the rest of pool_v2"
for try in 1 2 3; do
  SET=$S/pool_v2_nemotron/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $O/pool_v2_vllm.jsonl 128 2>&1 | tee -a $O/pool_v2_vllm.log | tail -n 1
done
tmux kill-window -t htr:parse2
log "pool_v2 read: $(wc -l < $O/pool_v2_vllm.jsonl)"
