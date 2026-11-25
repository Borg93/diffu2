"""The Nemotron 3.5 Parse run's best checkpoint reading the 200 test pages the Unlimited-OCR 2 and LightOnOCR-3 reads drew
(same pages, by id), with its default task prompt, decoded as its generation config says (greedy, repetition penalty
1.1), at most MAX_NEW tokens; writes each reading's text (boxes, classes and table markup left out) beside the ground
truth's, as one shard.

    CUDA_VISIBLE_DEVICES=2 packages/htr_train/envs/nemotron_parse/.venv/bin/python runs/htr/verify/nemotron_parse_test_read.py CKPT SHARD SHARDS OUT [LIMIT]
"""
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MAX_NEW = 4000  # LightOnOCR-3's 3,000 plus room for the box and class tokens around each block
LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
NO_REPEAT = int(os.environ.get("NO_REPEAT_NGRAM", "0"))  # 0: off; else no n-gram of this many tokens twice in a reading
IDS = os.environ.get("IDS", "").split(",") if os.environ.get("IDS") else None
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")

ckpt, shard, shards, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
wanted = [l.strip() for l in open("runs/htr/verify/test_pages_200.txt", encoding="utf-8") if l.strip() and not l.startswith("#")]  # the 200 scored test pages
rows = [l for l in open("data/htr/nemotron_parse_pages.test.jsonl", encoding="utf-8").read().split("\n") if l]  # not splitlines: U+0085
metas = [json.loads(l) for l in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")]
index = {m["id"]: i for i, m in enumerate(metas)}
picked = [index[i] for i in sorted(IDS or wanted) if i in index][shard::shards][:limit]
processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModel.from_pretrained(ckpt, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()


def text(answer: str) -> str:
    return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())


with open(out, "w", encoding="utf-8") as f:
    for i in picked:
        row = json.loads(rows[i])
        prompt = row["messages"][0]["content"].replace("<image>", "")
        image = Image.open(row["images"][0]).convert("RGB")
        inputs = processor(images=[image], text=prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW, no_repeat_ngram_size=NO_REPEAT)
        reading = processor.batch_decode(ids, skip_special_tokens=False)[0]
        reading = reading.split(prompt, 1)[-1].replace("</s>", "").replace("<s>", "").replace("<pad>", "")
        f.write(json.dumps({"id": metas[i]["id"], "subset": metas[i]["collection"],
                            "text": text(row["messages"][1]["content"]), "prediction": text(reading), "raw": reading,
                            "tokens": int(ids.shape[1]), "seconds": round(time.perf_counter() - started, 2)},
                           ensure_ascii=False) + "\n")
        f.flush()
