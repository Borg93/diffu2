#!/bin/bash
echo "retired 20 Nov 2026: this script starts jobs on GPU 1, which is no longer ours; see runs/htr/resume_after_mount.sh"; exit 1
# 18 Nov, in place of after_trusted2.sh: what the GPUs do when the training queue (tmux `trusted`, `aftertrusted`) has
# ended. (1) The ground-truth scans without patch vectors are embedded and get their columns. (2) Our best reader
# (nemotron-parse-more) reads the 139,937 trusted-cut pages through vLLM with the DFlash2 draft, one server a GPU
# (CURATION E50; 3.2 pages a second on a shared GPU against 0.5 with pool_read.py), for the label filter (E12).
# Resumable; stop it for the next training by closing its windows (tmux kill-window -t htr:parse1, parse2, aftertrusted3).
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
for g in 1 2; do
  tmux new-window -d -t htr -n parse$g "GPU=$g PORT=801$g FRACTION=0.85 SEQS=128 bash runs/htr/parse_serve.sh 2>&1 | tee runs/htr/verify/parse_serve_$g.log"
done
for g in 1 2; do until curl -s -m 2 http://127.0.0.1:801$g/v1/models > /dev/null 2>&1; do sleep 10; tmux list-windows -t htr -F '#W' | grep -q -x parse$g || { log "FAILED server $g"; exit 1; }; done; done
log "both servers up: reading the trusted pages"
SET=$S/trusted_v1_nemotron/pages URLS=http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1 uv run --with httpx python runs/htr/verify/parse_vllm_pool_read.py $O/trusted_v1_vllm.jsonl 192 2>&1 | tee $O/trusted_v1_vllm.log | tail -n 2
tmux kill-window -t htr:parse1; tmux kill-window -t htr:parse2
log "trusted pages read: $(wc -l < $O/trusted_v1_vllm.jsonl)"
