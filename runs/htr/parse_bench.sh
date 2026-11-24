#!/bin/bash
# 24 Nov: how fast the reader reads under each serving setting (asked: is 6.8 pages a second all there is), on GPU 2
# alone: the pool read is stopped for it and started again at the end with the fastest setting. The same 2,000 pages of
# pool_v5 for each (a fresh server each time, so no page is in its cache). Results: runs/htr/verify/parse_bench.tsv
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; B=$S/parse_bench; R=runs/htr/verify/parse_bench.tsv
log() { echo "[$(date '+%F %T')] $*"; }
tmux kill-session -t =poolread 2>/dev/null; tmux kill-window -t htr:parse2 2>/dev/null; sleep 20
[ -s $R ] || printf 'draft\tseqs\tpages\tseconds\tpages_per_s\ttokens\tlength_stops\n' > $R
for cfg in "16 128" "0 128" "4 128" "8 128" "0 256" "4 256"; do
  set -- $cfg; D=$1; Q=$2; out=$B/read_d${D}_s${Q}.jsonl; rm -f $out
  tmux new-window -d -t htr -n parsebench "GPU=2 PORT=8012 FRACTION=0.85 SEQS=$Q DRAFT=$D bash runs/htr/parse_serve.sh $S/serve/parse-selftrain2 2>&1 | tee runs/htr/verify/parse_bench_d${D}_s${Q}.log"
  until curl -s -m 2 http://127.0.0.1:8012/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parsebench || { log "FAILED server draft $D seqs $Q"; continue 2; }; done
  t0=$(date +%s.%N)
  SET=$B/pages URLS=http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $out $Q 2>&1 | tail -n 1
  t1=$(date +%s.%N)
  python3 - $out $D $Q $t0 $t1 >> $R <<'P'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]; s = float(sys.argv[5]) - float(sys.argv[4])
print(sys.argv[2], sys.argv[3], len(rows), round(s), round(len(rows) / s, 2), sum(r["tokens"] for r in rows), sum(r["finish"] == "length" for r in rows), sep="\t")
P
  log "draft $D seqs $Q: $(tail -n 1 $R)"
  tmux kill-window -t htr:parsebench; sleep 20
done
best=$(tail -n +2 $R | sort -t$'\t' -k5 -g -r | head -n 1); set -- $best
log "fastest: draft $1, $2 in flight ($5 pages a second): the pool read goes on with it"
tmux new-session -d -s poolread "SEQS=$2 DRAFT=$1 bash runs/htr/pool_rest_read.sh 2>&1 | tee -a runs/htr/pool_rest_read.log"
