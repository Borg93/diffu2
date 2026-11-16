"""The defined cut-or-not question (E32) asked of every scan the first form calls an opening, into cut_or_not.lance.

Appends (image, answer) in batches; resumable. The answer joins scans.lance as ``vl_cut`` by to_lance.py's helper
once the run has ended.

    uv run python runs/htr/curate/prototypes/vl_cut_all.py [PORT]
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import pyarrow as pa

import vl_cut_or_not as question

HOME = Path("runs/htr/curate"); OUT = HOME / "cut_or_not.lance"; BATCH, WORKERS = 2000, 64
ROW = pa.schema([("image", pa.utf8()), ("answer", pa.utf8()), ("error", pa.utf8())])

if __name__ == "__main__":
    openings = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image"], filter="vl_layout IN ('opening_both_pages_written', 'opening_one_page_empty')").column("image").to_pylist()
    done = set(lance.dataset(str(OUT)).to_table(columns=["image"], filter="error IS NULL").column("image").to_pylist()) if OUT.exists() else set()
    todo = [image for image in openings if image not in done]; print(len(openings), "openings,", len(todo), "to ask", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(todo), BATCH):
            rows = list(pool.map(question.ask, [(image, "definitions", []) for image in todo[start:start + BATCH]]))
            table = pa.table({"image": [r["image"] for r in rows], "answer": [r.get("answer") if isinstance(r.get("answer"), str) else None for r in rows], "error": [r.get("error") for r in rows]}, schema=ROW)
            lance.write_dataset(table, str(OUT), mode="append" if OUT.exists() else "create")
            print(start + len(rows), "of", len(todo), flush=True)
    lance.dataset(str(OUT)).optimize.compact_files(); print("done", flush=True)
