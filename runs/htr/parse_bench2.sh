#!/bin/bash
# 24 Nov: after parse_bench.sh. Two more measures on GPU 2 alone, the pool read stopped for them:
#  1. the image stage alone: every request stops after one token (MAX_NEW=1), no draft: pages a second the encoder and
#     the prefill give when nothing is written;
#  2. the FlashInfer attention backend (this GPU is SM120: vLLM refuses FlashAttention 3 and 4 on it and falls back to 2).
# Then the pool read goes on with the fastest full setting. Results: runs/htr/verify/parse_bench.tsv (a column more: note)
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; B=$S/parse_bench; R=runs/htr/verify/parse_bench.tsv
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "fastest" runs/htr/parse_bench.log; do sleep 30; done
sleep 60; tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 20
run() {  # name draft seqs max_new attn
  out=$B/read_$1.jsonl; rm -f $out
  tmux new-window -d -t htr -n parsebench "ATTN='$5' GPU=2 PORT=8012 FRACTION=0.85 SEQS=$3 DRAFT=$2 bash runs/htr/parse_serve.sh $S/serve/parse-selftrain2 2>&1 | tee runs/htr/verify/parse_bench_$1.log"
  until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parsebench || { log "FAILED server $1"; return; }; done
  t0=$(date +%s.%N); MAX_NEW=$4 SET=$B/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $out $3 2>&1 | tail -n 1; t1=$(date +%s.%N)
  python3 - $out $2 $3 $t0 $t1 $1 >> $R <<'P'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; s = float(sys.argv[5]) - float(sys.argv[4])
print(sys.argv[2], sys.argv[3], len(rows), round(s), round(len(rows) / s, 2), sum(r["tokens"] for r in rows), sum(r["finish"] == "length" for r in rows), sys.argv[6], sep="\t")
P
  log "$1: $(tail -n 1 $R)"; tmux kill-window -t htr:parsebench; sleep 20
}
run image_only 0 128 1 ''
run flashinfer_d8 8 128 3000 '{"backend": "FLASHINFER"}'
run flashinfer_d0 0 128 3000 '{"backend": "FLASHINFER"}'
best=$(tail -n +2 $R | grep -v image_only | sort -t$'\t' -k5 -g -r | head -n 1); set -- $best
A=''; case "$8" in flashinfer*) A='{"backend": "FLASHINFER"}';; esac
log "fastest: draft $1, $2 in flight, ${8:-flash-attn 2} ($5 pages a second): the pool read goes on with it"
tmux new-session -d -s poolread "ATTN='$A' SEQS=$2 DRAFT=$1 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"
