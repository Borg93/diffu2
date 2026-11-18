#!/bin/bash
# 18 Nov: the GPUs after the trusted pages' read (after_trusted3.sh), until Friday. Our best reader reads the
# million pages of pool_v2 through vLLM (DFlash2 draft, repetition detector), one server a GPU: every page then has
# its disagreement with its old label (the label filter for the next pretraining) and the reader's own answer (the
# data a draft of our own trains on, E52). Resumable; stop it with tmux kill-window on parse1, parse2, aftertrusted4.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'trusted|aftertrusted|aftertrusted3'; do sleep 60; done
until [ -s $S/pool_v2_nemotron/BUILT ]; do sleep 120; tmux list-windows -t htr -F '#W' | grep -q -x poolv2build || { [ -s $S/pool_v2_nemotron/BUILT ] || { log "FAILED: the set was not built"; exit 1; }; }; done
log "pool_v2 built ($(cat $S/pool_v2_nemotron/BUILT) pages): starting the servers"
for g in 1 2; do
  tmux kill-window -t htr:parse$g 2>/dev/null
  tmux new-window -d -t htr -n parse$g "GPU=$g PORT=801$g FRACTION=0.85 SEQS=128 bash runs/htr/parse_serve.sh 2>&1 | tee runs/htr/verify/parse_serve_$g.log"
done
for g in 1 2; do until curl -s -m 2 http://127.0.0.1:801$g/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parse$g || { log "FAILED server $g"; exit 1; }; done; done
log "both servers up: reading pool_v2"
for try in 1 2 3; do  # a page a failed request left unread is taken by the next pass
  SET=$S/pool_v2_nemotron/pages URLS=http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $O/pool_v2_vllm.jsonl 192 2>&1 | tee -a $O/pool_v2_vllm.log | tail -n 1
done
tmux kill-window -t htr:parse1; tmux kill-window -t htr:parse2
log "pool_v2 read: $(wc -l < $O/pool_v2_vllm.jsonl)"
