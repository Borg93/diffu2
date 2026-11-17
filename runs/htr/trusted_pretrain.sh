#!/bin/bash
# 17 Nov, the user's yes (CURATION E38): Nemotron 3.5 Parse pretrained on the pool pages whose cut is trusted, then
# fine-tuned on the GT as nemotron-parse-all (2 epochs) and as nemotron-parse-more (2 more), the 200 test pages read
# with the 10-token no-repeat guard after each. Waits for the embedding of the described scans (tmux embed2a, embed2b).
cd /home/gabriel/Desktop/diffu2 || exit 1
L=packages/htr_train/lion_data_prep; S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -E '^embed2[ab]$'; do sleep 60; done
log "embedding ended"
uv run python $P/embeddings_to_lance.py $S/embed_described2 $S/embed_described2a $S/embed_described2b | tail -n 1 || { log "FAILED embeddings"; exit 1; }
uv run python $P/cut_head.py || { log "FAILED head"; exit 1; }
uv run python $P/stage_trusted.py $S/trusted_v1 || { log "FAILED stage"; exit 1; }
uv run diffu-reader-set --xml $S/trusted_v1 --model nemotron-parse --cuts $S/trusted_v1/cuts.json --split-from-gt-test-lists --workers 24 --out $S/trusted_v1_nemotron/pages.jsonl || { log "FAILED set"; exit 1; }
log "set built: $(wc -l < $S/trusted_v1_nemotron/pages.jsonl) pages"
for pid in $(cat runs/htr/curate/pool_read.pids 2>/dev/null); do kill "$pid" 2>/dev/null && log "pool read $pid stopped (resumable)"; done
sleep 30
read_test() {  # NAME of the run, folder of the read
  local R=runs/htr/$1 O=runs/htr/nemotron_parse/$2 BEST; mkdir -p "$O"
  BEST=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['best_model_checkpoint'] or '')" $(ls -d $R/output/checkpoint-*/trainer_state.json | sort -V | tail -1))
  [ "$3" = last ] && BEST=$(ls -d $R/output/checkpoint-* | sort -V | tail -1)
  log "$1: reading the test pages with $BEST"
  for s in 0 1; do
    NO_REPEAT_NGRAM=10 CUDA_VISIBLE_DEVICES=$((s + 1)) timeout 2h packages/htr_train/envs/nemotron_parse/.venv/bin/python runs/htr/verify/nemotron_parse_test_read.py "$BEST" $s 2 $O/shard$s.jsonl > $O/shard$s.log 2>&1 &
  done
  wait
  cat $O/shard*.jsonl > $O/pages.jsonl; log "read $(wc -l < $O/pages.jsonl) of 200 test pages"
  COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $O/pages.jsonl > $O/cer.txt 2>&1; log "scored: $O/cer.txt"
}
GPUS=1,2 bash $L/train_pages.sh $L/configs/nemotron_parse_trusted_pretrain.yaml nemotron-parse-trusted-pretrain || { log "pretraining failed: no fine-tune"; exit 1; }
GPUS=1,2 bash $L/train_pages.sh $L/configs/nemotron_parse_trusted_finetune.yaml nemotron-parse-trusted-finetune || { log "fine-tune failed"; exit 1; }
read_test nemotron-parse-trusted-finetune test_read_trusted_norepeat10 best
GPUS=1,2 bash $L/train_pages.sh $L/configs/nemotron_parse_trusted_more.yaml nemotron-parse-trusted-more || { log "more failed"; exit 1; }
read_test nemotron-parse-trusted-more test_read_trusted_more_norepeat10 last
log "all done"
