"""Base MiniCPM-OCR on whole regions against the same regions' lines: 300 text regions drawn (seed 0) from the test
side of the region set (tables left out: they take another prompt), and every test line of those regions."""

import json
import random
from pathlib import Path

from diffu_htr_train.evaluate.lines import EvalLine, write_lines

D = Path("data/htr")
OUT = Path("runs/htr/regiontest")


def rows(stem: str) -> list[tuple[dict, dict]]:
    samples = (D / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
    metas = (D / f"{stem}.meta.jsonl").read_text(encoding="utf-8").splitlines()
    return [(json.loads(s), json.loads(m)) for s, m in zip(samples, metas, strict=True)]


def line(sample: dict, meta: dict) -> EvalLine:
    answer = next(m["content"] for m in sample["messages"] if m["role"] == "assistant")
    return EvalLine(id=meta["id"], subset=meta["collection"], image=sample["images"][0], text=answer)


regions = [(s, m) for s, m in rows("minicpm_regions.test") if m.get("kind") != "table"
           and "Text Recognition:" in s["messages"][0]["content"]]
lines = rows("minicpm_lines_scaled.test")
by_region: dict[str, list[tuple[dict, dict]]] = {}
for s, m in lines:
    by_region.setdefault(m["id"].rsplit("_", 1)[0], []).append((s, m))
with_lines = [(s, m) for s, m in regions if m["id"] in by_region]
picked = random.Random(0).sample(with_lines, min(300, len(with_lines)))
write_lines([line(s, m) for s, m in picked], OUT / "regions.jsonl")
picked_lines = [line(s, m) for _, rm in picked for s, m in sorted(by_region[rm["id"]], key=lambda p: p[1]["line"])]
write_lines(picked_lines, OUT / "lines.jsonl")
print(f"{len(regions)} text regions on test pages, {len(with_lines)} with their lines kept; "
      f"{len(picked)} drawn, {len(picked_lines)} lines")
