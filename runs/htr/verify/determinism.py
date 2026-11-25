"""The same 200 drawn test lines read twice by the card-served reader, 16 at a time and one at a time: how often the
reader disagrees with itself, and the CER of each run and of the queue's readings, against the references."""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from check_reader import read  # noqa: E402 (beside this file)
from diffu_htr_train.evaluate.cer import edits, whitespace

client = OpenAI(base_url=f"{sys.argv[1]}/v1", api_key="-")
lines = [json.loads(line) for line in open("data/htr/eval/test5000.jsonl", encoding="utf-8")]
queue = {r["id"]: r["prediction"] for r in map(json.loads, open("runs/htr/final/test_minicpm.jsonl", encoding="utf-8"))}
draw = random.Random(0).sample(lines, 200)
with ThreadPoolExecutor(16) as pool:
    batched = [text for text, _ in pool.map(lambda row: read(client, row["image"], 512), draw)]
single = [read(client, row["image"], 512)[0] for row in draw]


def cer(texts: list[str]) -> float:
    return sum(edits(whitespace(r["text"]), whitespace(t)) for r, t in zip(draw, texts, strict=True)) / sum(
        len(whitespace(r["text"])) for r in draw)


print(f"16 at a time vs one at a time: {sum(a != b for a, b in zip(batched, single))} of 200 differ")
print(f"one at a time vs the queue (64 at a time): {sum(a != queue[r['id']] for r, a in zip(draw, single))} of 200 differ")
print(f"corpus CER: 16 at a time {cer(batched):.2%}, one at a time {cer(single):.2%}, "
      f"queue {cer([queue[r['id']] for r in draw]):.2%}")
