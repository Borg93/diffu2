#!/bin/bash
# 23 Nov: what the pretraining's last checkpoint (the fine-tune's start) read on each dev page, to set beside the
# fine-tune's checkpoints page by page. Waits for the watcher's re-read of steps 250 and 500 (one reader at a time on GPU 2).
cd /home/gabriel/Desktop/diffu2 || exit 1
R=runs/htr/nemotron-parse-selftrain-pretrain
until [ -f runs/htr/nemotron-parse-selftrain-finetune2/dev_reads/step-500.jsonl ]; do sleep 30; done
mkdir -p $R/dev_reads
CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/nemotron_parse/.venv/bin/python - <<'P'
from pathlib import Path
from diffu_htr_train.nemotron_parse.checkpoint_cer import dev_pages, drawn, page_reads, read
run = Path("runs/htr/nemotron-parse-selftrain-pretrain")
pages = drawn(dev_pages(Path("data/htr/nemotron_parse_pages.dev.jsonl")), 30, 0)
reads = page_reads(pages, read(run / "output" / "checkpoint-4319", pages))
(run / "dev_reads" / "step-4319.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in reads), "utf-8")
print("done", sum(r.edits for r in reads) / sum(r.chars for r in reads))
P
