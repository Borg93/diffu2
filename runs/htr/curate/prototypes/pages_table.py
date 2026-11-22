"""The pages as built for training, a row a page (CURATION E67, step 1): ``runs/htr/curate/train_pages.lance``.

    NUMPY_MADVISE_HUGEPAGE=0 ARROW_DEFAULT_MEMORY_POOL=system uv run --with rapidfuzz python \
        runs/htr/curate/prototypes/pages_table.py

A row holds the page's id (the set builder's), its set, scan, side, where the scan was cut (the fold's place as a share
of the width), the page image, the old label (the training target as built, and its plain text), and, where our reader
has read the page, its text, how the reading ended and the disagreement with the old label. ``image`` is the scan's key
in scans.lance, so every scan column reaches a page by it. The older pages.lance (the first cut) is left as it is.
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import json, re, unicodedata
from pathlib import Path

import lance, pyarrow as pa
from rapidfuzz.distance import Levenshtein

HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr"); OUT = HOME / "train_pages.lance"
SETS = ("trusted_v1", "pool_v2", "pool_v3", "pool_v4", "pool_v5"); TAG = re.compile(r"<[^>]+>"); BATCH = 50_000
SCHEMA = pa.schema([("page", pa.utf8()), ("set", pa.utf8()), ("image", pa.utf8()), ("pool", pa.utf8()), ("volume", pa.utf8()), ("scan", pa.utf8()),
                    ("side", pa.utf8()), ("fold", pa.float32()), ("page_image", pa.utf8()), ("target", pa.utf8()), ("label", pa.utf8()),
                    ("label_chars", pa.int32()), ("label_lines", pa.int32()), ("read_nemotron", pa.utf8()), ("read_finish", pa.utf8()),
                    ("read_chars", pa.int32()), ("read_disagreement", pa.float32())])


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def flat(text: str) -> str:
    """Line breaks and runs of spaces as one space: the reading has no line breaks, so the comparison must not count them."""
    return " ".join(text.split())


def rows(name: str, scans: dict[str, tuple[str, str]]):
    folder = S / f"{name}_nemotron"
    cuts = json.loads((S / name / "cuts.json").read_text()) if (S / name / "cuts.json").exists() else {}
    reads = {}
    read_file = HOME / f"pool_read/{name}_vllm.jsonl"
    if read_file.exists():
        for line in open(read_file, encoding="utf-8"):
            r = json.loads(line); reads[r["id"]] = (r["prediction"], r.get("finish"))
    with open(folder / "pages.jsonl", encoding="utf-8") as targets, open(folder / "pages.meta.jsonl", encoding="utf-8") as metas:
        for target_line, meta_line in zip(targets, metas, strict=True):
            m = json.loads(meta_line); target = json.loads(target_line)["messages"][1]["content"]
            scan = m["scan"].rsplit(".", 1)[0]; image, pool = scans.get(scan, (None, None)); label = clean(TAG.sub("", target))
            read, finish = reads.get(m["id"], (None, None)); read = None if read is None else clean(read)
            yield {"page": m["id"], "set": name, "image": image, "pool": pool, "volume": m["volume"], "scan": scan, "side": m["side"],
                   "fold": cuts.get(scan), "page_image": str(folder / "pages_pages" / f"{m['id']}.jpg"), "target": target, "label": label,
                   "label_chars": len(label), "label_lines": label.count("\n") + 1 if label else 0, "read_nemotron": read, "read_finish": finish,
                   "read_chars": None if read is None else len(read),
                   "read_disagreement": None if read is None else Levenshtein.distance(flat(label), flat(read)) / max(1, len(flat(label)))}


if __name__ == "__main__":
    assert not OUT.exists(), f"{OUT} exists: remove it yourself to build again"
    t = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["scan", "image", "pool"], filter="pool != 'gt'").to_pandas()
    twice = set(t.scan[t.scan.duplicated()]); scans = {s: (i, p) for s, i, p in zip(t.scan, t.image, t.pool) if s not in twice}
    print(len(scans), "scans keyed;", len(twice), "scan names held twice (their pages get no key)", flush=True)
    seen: set[str] = set()
    for name in SETS:
        held, n, again = [], 0, 0
        for row in rows(name, scans):
            if row["page"] in seen:  # a page of an earlier set (trusted_v1's pages are also pool pages): the first set keeps it
                again += 1; continue
            seen.add(row["page"]); held.append(row); n += 1
            if len(held) == BATCH:
                lance.write_dataset(pa.Table.from_pylist(held, schema=SCHEMA), str(OUT), mode="append"); held = []
        if held: lance.write_dataset(pa.Table.from_pylist(held, schema=SCHEMA), str(OUT), mode="append")
        print(name, n, "pages written;", again, "held by an earlier set", flush=True)
    d = lance.dataset(str(OUT)); d.optimize.compact_files(); d = lance.dataset(str(OUT))
    for column, kind in (("page", "BTREE"), ("image", "BTREE"), ("set", "BITMAP"), ("pool", "BITMAP"), ("read_disagreement", "BTREE")): d.create_scalar_index(column, kind)
    d.tags.create("built-22nov", d.version); print("done:", d.count_rows(), "rows, version", d.version, flush=True)
