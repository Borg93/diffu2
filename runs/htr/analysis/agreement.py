"""How good a TrOCR label is where our reader agrees with it: the lines both read, split by how far apart the two
readings are (CER between them), each group's share of the lines and TrOCR's CER against the human reference."""

import json
import sys
from pathlib import Path

from diffu_htr_train.evaluate.cer import edits, whitespace


def rows(path: Path) -> dict[str, dict]:
    return {row["id"]: row for row in map(json.loads, path.open(encoding="utf-8"))}


ours, trocr = rows(Path(sys.argv[1])), rows(Path(sys.argv[2]))
shared = [i for i in ours if i in trocr]
bands = [("identical", 0.0), ("<= 2% apart", 0.02), ("<= 5% apart", 0.05), ("<= 10% apart", 0.10), ("all", 9e9)]
lines = []
for i in shared:
    ref, a, b = whitespace(ours[i]["text"]), whitespace(ours[i]["prediction"]), whitespace(trocr[i]["prediction"])
    apart = edits(a, b) / max(len(b), 1)
    lines.append((apart, edits(ref, b), len(ref), ref, a, b, ours[i]["subset"]))
total_edits = sum(line[1] for line in lines)
print(f"{sys.argv[1]}: {len(lines)} lines both read; TrOCR corpus CER {total_edits / sum(l[2] for l in lines):.2%}")
for name, limit in bands:
    kept = [line for line in lines if line[0] <= limit]
    errs, chars = sum(line[1] for line in kept), sum(line[2] for line in kept)
    wrong = sum(1 for line in kept if line[1])
    print(
        f"  {name:13} {len(kept) / len(lines):6.1%} of lines | TrOCR label CER {errs / max(chars, 1):6.2%} | "
        f"lines with any error {wrong / max(len(kept), 1):6.1%} | keeps {errs / total_edits:6.1%} of TrOCR's errors"
    )
both_wrong = [line for line in lines if line[0] == 0 and line[1]]
print(f"  identical readings that differ from the reference: {len(both_wrong)}")
for apart, err, n, ref, a, b, subset in both_wrong[:6]:
    print(f"    [{subset[:28]}] ref {ref[:70]!r}\n    {'':30} both {a[:70]!r}")
