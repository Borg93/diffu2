"""The Unlimited-OCR 2 run's best checkpoint reading a seeded sample of test pages with the model's own inference
(single page, gundam tiles, its own sparse-attention routing), its answers capped at MAX_NEW tokens; writes each
reading and the ground truth's text, as one shard of the sample.

    CUDA_VISIBLE_DEVICES=1 packages/htr_train/envs/uocr2/.venv/bin/python runs/htr/verify/uocr2_test_read.py CKPT SHARD SHARDS OUT
"""
import json
import random
import re
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

SAMPLE, MAX_NEW = 200, 3000
ckpt, shard, shards, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rows = open("data/htr/uocr2_pages.test.jsonl", encoding="utf-8").read().splitlines()
metas = open("data/htr/uocr2_pages.test.meta.jsonl", encoding="utf-8").read().splitlines()
picked = sorted(random.Random(0).sample(range(len(rows)), SAMPLE))[shard::shards]
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()


def text(answer: str) -> str:
    """The blocks' text alone, whitespace collapsed (labels and boxes left out)."""
    return " ".join(re.sub(r"<BLOCK>[a-z_]+\[[^\]]*\]<FORK>", "\n", answer.replace("<PAGE>", "")).split())


with open(out, "w", encoding="utf-8") as f:
    for i in picked:
        row = json.loads(rows[i])
        started = time.perf_counter()
        result = model.infer(tok, image_file=row["images"][0], output_path="/tmp", save_results=False, eval_mode=True,
                             max_new_tokens=MAX_NEW)
        reading = result[0] if isinstance(result, tuple) else result
        f.write(json.dumps({"id": json.loads(metas[i])["id"], "subset": json.loads(metas[i])["collection"],
                            "text": text(row["messages"][1]["content"]), "prediction": text(reading or ""),
                            "raw": reading, "seconds": round(time.perf_counter() - started, 2)}, ensure_ascii=False) + "\n")
        f.flush()
