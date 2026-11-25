"""Whole regions against the same regions read line by line: each region's reference against its whole reading, and
against its lines' readings joined by line breaks (its reference rebuilt from its lines' references, so a line the
set left out is out on both sides); OpenBMB's corpus CER (NFC, ends trimmed) and the per-region mean."""

import json
from collections import defaultdict
from pathlib import Path

from diffu_htr_train.evaluate.cer import edits, nfc_trimmed

O = Path("runs/htr/regiontest")
regions = {r["id"]: r for r in map(json.loads, (O / "read_regions.jsonl").read_text().splitlines())}
lines = defaultdict(list)
for r in map(json.loads, (O / "read_lines.jsonl").read_text().splitlines()):
    lines[r["id"].rsplit("_", 1)[0]].append(r)


def rates(pairs):
    pairs = [(nfc_trimmed(a), nfc_trimmed(b)) for a, b in pairs]
    total = sum(edits(a, b) for a, b in pairs)
    chars = sum(len(a) for a, _ in pairs)
    mean = sum(edits(a, b) / max(1, len(a)) for a, b in pairs) / len(pairs)
    return total / chars, mean


whole = [(r["text"], r["prediction"]) for r in regions.values()]
joined = [("\n".join(x["text"] for x in lines[i]), "\n".join(x["prediction"] for x in lines[i])) for i in regions]
cut = sum(r.get("finished") == "length" for r in regions.values())
for name, pairs in (("whole regions", whole), ("their lines, joined", joined)):
    corpus, mean = rates(pairs)
    print(f"{name:22} corpus CER {corpus:6.2%}   mean CER a region {mean:6.2%}   ({len(pairs)} regions)")
print(f"regions cut off at 4,096 tokens: {cut}")
for name in ("regions", "lines"):
    print(f"{name}: " + (O / f"read_{name}.speed.json").read_text().replace("\n", " "))
