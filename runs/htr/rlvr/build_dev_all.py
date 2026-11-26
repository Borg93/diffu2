"""All 974 dev pages (5 whole volumes no training set holds) as a bundle for ``eval_pilot.sh`` (CURATION E72).

The monitor split of ``data_v2`` is a draw of 115 of them (at most 30 a volume); with 115 pages a change under about
0.2 to 0.3 points of CER cannot be told from chance. This writes ``/mnt/scratch/gabriel/htr/rlvr/data_dev_all`` with
``monitor.jsonl`` = every dev page, in the rows ``eval_read.py`` and ``eval_score.py`` read (``id``, ``image`` under
``images/``, ``prompt``, ``target``, ``collection``, ``volume``, ``scan``, ``split``, and ``target_tokens`` (with EOS) and ``has_table`` as
``build_dataset.py`` measures them, which the scorer's page kinds read); no train or test split.

    /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/build_dev_all.py
"""

import json
import os
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000"
SET = Path("data/htr/nemotron_parse_pages.dev.jsonl")
META = Path("data/htr/nemotron_parse_pages.dev.meta.jsonl")
OUT = Path("/mnt/scratch/gabriel/htr/rlvr/data_dev_all")


def lines(path: Path) -> list[str]:
    """The file's lines, split at newlines alone (``str.splitlines`` also splits at U+0085 inside a label)."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


def main() -> None:
    rows, metas = lines(SET), lines(META)
    if len(rows) != len(metas):
        raise SystemExit(f"{len(rows)} pages, {len(metas)} lines of meta")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    written = []
    for line, meta_line in zip(rows, metas, strict=True):
        row, meta = json.loads(line), json.loads(meta_line)
        source = Path(row["images"][0])
        link = OUT / "images" / f"{meta['id']}{source.suffix}"
        if not link.exists():
            os.symlink(source, link)
        written.append({
            "id": meta["id"],
            "image": f"images/{link.name}",
            "prompt": row["messages"][0]["content"].replace("<image>", ""),
            "target": row["messages"][1]["content"],
            "collection": meta["collection"],
            "volume": meta["volume"],
            "scan": meta["scan"],
            "side": meta.get("side"),
            "context_id": f"{meta['collection']}/{meta['volume']}/{meta['scan']}",
            "split": "monitor",
            "target_tokens": len(tokenizer.encode(row["messages"][1]["content"], add_special_tokens=False)) + 1,
            "has_table": "\\begin{tabular}" in row["messages"][1]["content"],
        })
    (OUT / "monitor.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in written), "utf-8")
    print(len(written), "dev pages of", len({r["volume"] for r in written}), "volumes ->", OUT / "monitor.jsonl")


if __name__ == "__main__":
    main()
