#!/bin/bash
# 18 Nov: what the GPUs do when the training queue (tmux `trusted`) and its follow-on (`aftertrusted`) have ended,
# until the next training is decided. (1) The ground-truth scans without patch vectors are embedded as the pools were
# and get the per-scan columns. (2) Our best reader (nemotron-parse-more, 6.73 %, guard 10) reads the 139,937
# trusted-cut pages, four shards, two a GPU, so every page has its disagreement with its old label (E12 on the trusted
# pages: the label filter). Both are resumable and may be stopped at any time: stop the windows' processes by PID.
cd /home/gabriel/Desktop/diffu2 || exit 1
S=/mnt/scratch/gabriel/htr; P=runs/htr/curate/prototypes; O=runs/htr/curate/pool_read
log() { echo "[$(date '+%F %T')] $*"; }
while tmux list-windows -t htr -F '#W' | grep -q -x -E 'trusted|aftertrusted|embedpool0|embedpool1'; do sleep 60; done
log "the training queue and the pool embedding have ended"
uv run python - <<'PY'
import lance
held = lance.dataset("runs/htr/curate/scans.lance").to_table(columns=["image"], filter="pool = 'gt' AND cut_class IS NULL").column("image").to_pylist()
open("/mnt/scratch/gabriel/htr/embed_gt_todo.txt", "w", encoding="utf-8").write("\n".join(held) + "\n"); print(len(held), "ground-truth scans to embed")
PY
IMAGES=$S/embed_gt_todo.txt CUDA_VISIBLE_DEVICES=1 WORKERS=24 uv run --with sentencepiece python $P/embed_pool.py $S/embed_gt 0 1 2>&1 | grep -v -i 'warn\|Loading weights\|return index' | tail -n 3
PYTHONPATH=$P flock runs/htr/curate/scans.lock uv run python $P/pool_columns.py apply $S/embed_gt 2>&1 | grep -v -i warn | tail -n 2
log "ground-truth scans embedded and given their columns"
for s in 0 1 2 3; do
  SET=$S/trusted_v1_nemotron/pages NO_REPEAT_NGRAM=10 CUDA_VISIBLE_DEVICES=$((1 + s % 2)) packages/htr_train/envs/nemotron_parse/.venv/bin/python \
    runs/htr/verify/pool_read.py runs/htr/nemotron-parse-more/output/checkpoint-3094 $s 4 $O/trusted_v1_$s.jsonl > $O/trusted_v1_$s.log 2>&1 &
done
wait
log "trusted pages read: $(cat $O/trusted_v1_?.jsonl | wc -l)"
