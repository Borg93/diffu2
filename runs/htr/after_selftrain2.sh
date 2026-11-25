#!/bin/bash
# 23 Nov: E60's fine-tune again from the pretraining's last checkpoint (the first one began from the checkpoint the dev
# loss kept, the worst by dev CER). Then the 200 test pages read with the checkpoint that is best by dev CER
# (RUN/dev_cer.jsonl, written by tmux cerwatch), scored and compared as the first self-training launcher did. GPU 2 only.
cd /home/gabriel/Desktop/diffu2 || exit 1
log() { echo "[$(date '+%F %T')] $*"; }
L=packages/htr_train/lion_data_prep; R=runs/htr/nemotron-parse-selftrain-finetune2
GPUS=2 bash $L/train_pages.sh $L/configs/nemotron_parse_selftrain_finetune2.yaml nemotron-parse-selftrain-finetune2 || { log "fine-tune failed"; exit 1; }
LAST=$(ls -d $R/output/checkpoint-* | sort -V | tail -n 1 | sed 's/.*checkpoint-//')
for _ in $(seq 90); do grep -q "\"step\":$LAST," $R/dev_cer.jsonl 2>/dev/null && break; sleep 60; done
BEST=$(python3 - "$R" <<'PY'
import json, os, sys
run = sys.argv[1]; rows = [json.loads(l) for l in open(f"{run}/dev_cer.jsonl")]
rows = [r for r in rows if os.path.exists(f"{run}/output/checkpoint-{r['step']}/model.safetensors")]
print(f"{run}/output/checkpoint-{min(rows, key=lambda r: r['cer'])['step']}")
PY
)
T=runs/htr/nemotron_parse/test_read_selftrain2_norepeat10; mkdir -p $T
log "reading the test pages with $BEST (best by dev CER)"
NO_REPEAT_NGRAM=10 CUDA_VISIBLE_DEVICES=2 timeout 3h packages/htr_train/envs/nemotron_parse/.venv/bin/python runs/htr/verify/nemotron_parse_test_read.py "$BEST" 0 1 $T/pages.jsonl > $T/read.log 2>&1
log "read $(wc -l < $T/pages.jsonl) of 200 test pages"
COLUMNS=200 packages/htr_train/envs/minicpm/.venv/bin/htr-eval cer $T/pages.jsonl > $T/cer.txt 2>&1
uv run --with rapidfuzz python runs/htr/curate/prototypes/compare_reads.py selftrain=test_read_selftrain2_norepeat10 oldlabels=test_read_trusted_norepeat10 none=test_read_norepeat10 > $T/compare.txt 2>&1
log "scored: $T/cer.txt, $T/compare.txt"
