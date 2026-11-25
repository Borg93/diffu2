"""What cutting a looping tail does to CER: a reading ending in a unit of 2+ characters repeated 3+ times, over 10+
characters, keeps one copy of it (so "450,000" or "..." are left alone)."""

import json
import re
import sys
from pathlib import Path

from diffu_htr_train.evaluate.cer import score, whitespace

LOOP = re.compile(r"(.{2,60}?)(?:\s*\1){2,}\s*$", re.DOTALL)
MIN_SPAN = 10


def guarded(text: str) -> str:
    """``text`` with a tail of 3+ repeats of one unit cut to a single copy."""
    match = LOOP.search(text)
    if match is None or len(match.group(0).strip()) < MIN_SPAN:
        return text
    return text[: match.start()] + match.group(1)


for path in map(Path, sys.argv[1:]):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    before = score([(r["text"], r["prediction"]) for r in rows], whitespace)
    after = score([(r["text"], guarded(r["prediction"])) for r in rows], whitespace)
    touched = [r for r in rows if guarded(r["prediction"]) != r["prediction"]]
    hurt = sum(
        1
        for r in touched
        if score([(r["text"], guarded(r["prediction"]))]).mean_cer > score([(r["text"], r["prediction"])]).mean_cer
    )
    print(
        f"{path.name:32} mean {before.mean_cer:.2%} -> {after.mean_cer:.2%}  corpus {before.corpus_cer:.2%} -> "
        f"{after.corpus_cer:.2%}  touched {len(touched)}, made worse {hurt}"
    )
    for r in touched[:3]:
        print(f"    ref {whitespace(r['text'])[-60:]!r}\n    got {whitespace(r['prediction'])[-60:]!r}")
