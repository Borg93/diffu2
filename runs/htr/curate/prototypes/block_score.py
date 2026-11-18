"""Nemotron Parse's blocks on the 200 test pages against the ground truth's: boxes, classes, reading order.

    uv run --with rapidfuzz python runs/htr/curate/prototypes/block_score.py [FOLDER below runs/htr/nemotron_parse]

A read block is matched to the truth block it overlaps most (one to one, greedily by IoU); found at IoU 0.5 or more.
The order is scored over the matched blocks: Kendall's tau of the read order against the truth's, a page with two or
more matched blocks.
"""
import json, re, sys
from itertools import combinations
import numpy as np
from rapidfuzz.distance import Levenshtein

BLOCK = re.compile(r"<x_([0-9.]+)><y_([0-9.]+)>(.*?)<x_([0-9.]+)><y_([0-9.]+)><class_([A-Za-z-]+)>", re.DOTALL)
def blocks(answer): return [((float(a), float(b), float(c), float(d)), " ".join(t.split()), k) for a, b, t, c, d, k in BLOCK.findall(answer)]
def iou(p, q):
    w = min(p[2], q[2]) - max(p[0], q[0]); h = min(p[3], q[3]) - max(p[1], q[1]); inter = max(w, 0) * max(h, 0)
    return inter / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter + 1e-9)

folder = sys.argv[1] if len(sys.argv) > 1 else "test_read_more_norepeat10"
reads = {r["id"]: r for r in map(json.loads, open(f"runs/htr/nemotron_parse/{folder}/pages.jsonl", encoding="utf-8"))}
rows = open("data/htr/nemotron_parse_pages.test.jsonl", encoding="utf-8").read().splitlines()
metas = [json.loads(l) for l in open("data/htr/nemotron_parse_pages.test.meta.jsonl", encoding="utf-8")]
truth = {m["id"]: blocks(json.loads(rows[i])["messages"][1]["content"]) for i, m in enumerate(metas) if m["id"] in reads}
n_truth = n_read = found = found75 = same_class = 0; ious, taus, cers, one, counts = [], [], [0, 0], 0, 0
for page, want in truth.items():
    got = blocks(reads[page]["raw"]); n_truth += len(want); n_read += len(got); counts += len(want) == len(got); one += len(want) == 1
    pairs = sorted(((iou(g[0], w[0]), i, j) for i, g in enumerate(got) for j, w in enumerate(want)), reverse=True); used_g, used_w, match = set(), set(), []
    for score, i, j in pairs:
        if score < 0.5: break
        if i in used_g or j in used_w: continue
        used_g.add(i); used_w.add(j); match.append((i, j)); ious.append(score); found += 1; found75 += score >= 0.75; same_class += got[i][2] == want[j][2]
        cers[0] += Levenshtein.distance(want[j][1], got[i][1]); cers[1] += len(want[j][1])
    if len(match) >= 2:
        match.sort(); order = [j for _, j in match]; pairs2 = list(combinations(range(len(order)), 2))
        taus.append(sum(1 if order[a] < order[b] else -1 for a, b in pairs2) / len(pairs2))
print(f"{len(truth)} pages; truth blocks {n_truth}, read blocks {n_read}; pages with one truth block {one}; pages with the truth's number of blocks {counts}")
print(f"truth blocks found at IoU 0.5: {found / n_truth:.3f} (at 0.75: {found75 / n_truth:.3f}) | read blocks that are a truth block: {found / n_read:.3f} | mean IoU of the found {np.mean(ious):.3f}")
print(f"class right on the found: {same_class / found:.3f} | CER inside the found blocks: {cers[0] / cers[1]:.4f}")
print(f"reading order, {len(taus)} pages with 2 or more found blocks: mean tau {np.mean(taus):.3f}; pages in exactly the truth's order {np.mean(np.array(taus) == 1):.3f}")
