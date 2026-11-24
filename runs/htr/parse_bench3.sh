#!/bin/bash
# 24 Nov: our own draft (E52, /mnt/scratch/gabriel/htr/draft/runs/v1) served: the same 2,000 pool_v5 pages as
# parse_bench.sh, 128 in flight, 8, 16 and 32 drafted tokens. Then the pool read goes on with the fastest setting of all.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; B=$S/parse_bench; R=runs/htr/verify/parse_bench.tsv; V=$S/serve/parse-selftrain2-draftv1
log() { echo "[$(date '+%F %T')] $*"; }
for D in 16 8 32; do
  out=$B/read_v1_d$D.jsonl; rm -f $out
  tmux new-window -d -t htr -n parsebench "GPU=2 PORT=8012 FRACTION=0.85 SEQS=128 DRAFT=$D bash runs/htr/parse_serve.sh $V 2>&1 | tee runs/htr/verify/parse_bench_v1_d$D.log"
  until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parsebench || { log "FAILED server v1 draft $D"; continue 2; }; done
  t0=$(date +%s.%N); SET=$B/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $out 128 2>&1 | tail -n 1; t1=$(date +%s.%N)
  python3 - $out $D 128 $t0 $t1 >> $R <<'P'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; s = float(sys.argv[5]) - float(sys.argv[4])
print(sys.argv[2], sys.argv[3], len(rows), round(s), round(len(rows) / s, 2), sum(r["tokens"] for r in rows), sum(r["finish"] == "length" for r in rows), "draft_v1", sep="\t")
P
  log "v1 draft $D: $(tail -n 1 $R)"; grep "Mean acceptance length" runs/htr/verify/parse_bench_v1_d$D.log | tail -n 1 | cut -c1-200
  tmux kill-window -t htr:parsebench; sleep 20
done
best=$(tail -n +2 $R | grep -v "image_only\|flashinfer" | sort -t$'\t' -k5 -g -r | head -n 1); set -- $best
SV=$S/serve/parse-selftrain2; [ "$8" = draft_v1 ] && SV=$V
log "fastest: draft $1 (${8:-NVIDIA's draft}), $5 pages a second: the pool read goes on with it"
tmux new-session -d -s poolread "SERVE=$SV SEQS=128 DRAFT=$1 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"
