"""vLLM's repetition detector replayed over saved readings (CURATION E51), as Reducto's note of 4 Sep 2026 does it.

    packages/htr_train/envs/vllm/.venv/bin/python runs/htr/curate/prototypes/repetition_replay.py

Loops: the readings that ran to their token cap (the unguarded vLLM reads of the test pages and of 700 trusted pages,
the first unguarded Transformers read). Good pages: the ground truth's own page targets (what a right reading looks
like, tables and ditto marks included) and the readings that ended by themselves. For each setting of
(min_pattern_size, max_pattern_size, min_count): the loops caught and the token at which, and the good pages it would
stop. The check is vLLM's (v1/core/sched/utils.py): the reading's tail is min_count copies of one n-gram.
"""
import json, random, sys
from pathlib import Path
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/mnt/scratch/gabriel/htr/serve/parse-more", trust_remote_code=True)
def ids(text): return tok(text, add_special_tokens=False)["input_ids"]
def rows(path): return [json.loads(l) for l in open(path, encoding="utf-8")]
def first_hit(tokens, lo, hi, count):
    """The token index at which the detector first fires, or None."""
    for end in range(lo * count, len(tokens) + 1):
        for n in range(lo, hi + 1):
            if n * count > end: break
            tail = tokens[end - n * count:end]
            if all(tail[k] == tail[k % n] for k in range(n, n * count)): return end
    return None

loops, good = [], []
for path, cap in (("runs/htr/nemotron_parse/test_read_vllm_base/pages.jsonl", 8744), (sys.argv[1] if len(sys.argv) > 1 else "", 3000), ("runs/htr/nemotron_parse/test_read/pages.jsonl", 4000)):
    if path and Path(path).exists():
        for r in rows(path): (loops if r["tokens"] >= cap - 1 else good).append(ids(r["raw"]))
truth = open("data/htr/nemotron_parse_pages.jsonl", encoding="utf-8").read().splitlines() if Path("data/htr/nemotron_parse_pages.jsonl").exists() else []
random.Random(0).shuffle(truth)
good += [ids(json.loads(l)["messages"][1]["content"]) for l in truth[:4000]]
print(len(loops), "looping readings;", len(good), "good pages (", min(4000, len(truth)), "of them ground-truth targets )", flush=True)
print(f"{'min':>4} {'max':>4} {'count':>5} | caught | median token | good pages stopped")
for lo, hi, count in ((1, 20, 10), (1, 40, 8), (1, 60, 6), (2, 60, 6), (3, 60, 5), (4, 80, 4), (4, 120, 4), (6, 120, 4), (8, 200, 3), (10, 200, 3), (1, 20, 20), (2, 60, 10), (4, 120, 6), (6, 200, 4)):
    hits = [first_hit(t, lo, hi, count) for t in loops]; caught = sorted(h for h in hits if h is not None)
    wrong = sum(first_hit(t, lo, hi, count) is not None for t in good)
    print(f"{lo:>4} {hi:>4} {count:>5} | {len(caught):>3}/{len(loops)} | {caught[len(caught) // 2] if caught else '-':>12} | {wrong}", flush=True)
