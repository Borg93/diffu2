"""What the page-level checks say, gathered per scan as columns of scans.lance (CURATION E58, E59).

    flock runs/htr/curate/scans.lock uv run --with rapidfuzz python runs/htr/curate/prototypes/page_checks_to_lance.py

For every scan whose pages are in a built set (trusted_v1, pool_v2 to pool_v5; the orientation network has seen all
of them, the reader those with a file in pool_read, pool_v3 as far as its read has come):
`pages_read`, `read_disagreement` (the reader's edits against the old label over the label's characters, the scan's
pages together; pages stopped as loops left out), `read_looped` (a page of it stopped as a loop), `turned` (a page of
it called turned with a probability over 0.9). Run again as the read goes on: it reads the reads' files as they stand.
"""
import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import json, subprocess, sys
from pathlib import Path

import lance, pandas as pd, pyarrow as pa

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate"); S = Path("/mnt/scratch/gabriel/htr")
parts = []
for name in ("trusted_v1", "pool_v2", "pool_v3", "pool_v4", "pool_v5"):
    was_read = (HOME / f"pool_read/{name}_vllm.jsonl").exists()
    if was_read: subprocess.run([sys.executable, str(HOME / "prototypes/read_disagreement.py"), str(HOME / f"pool_read/{name}_vllm.jsonl"), str(HOME / f"disagreement/{name}.parquet")], check=True, capture_output=True)
    scan_of = {}
    for line in open(S / f"{name}_nemotron/pages.meta.jsonl", encoding="utf-8"):
        meta = json.loads(line); scan_of[meta["id"]] = meta["scan"].rsplit(".", 1)[0]
    read = pd.read_parquet(HOME / f"disagreement/{name}.parquet")[["id", "label_chars", "edits", "finish"]] if was_read else pd.DataFrame({"id": pd.Series(dtype=str), "label_chars": pd.Series(dtype="int64"), "edits": pd.Series(dtype="int64"), "finish": pd.Series(dtype=object)})
    turn = pd.read_parquet(HOME / f"orientation/{name}.parquet")[["id", "turn", "p"]]
    pages = pd.DataFrame({"id": list(scan_of), "scan": list(scan_of.values())}).merge(read, on="id", how="left").merge(turn, on="id", how="left")
    parts.append(pages); print(name, len(pages), "pages;", int(pages.finish.notna().sum()), "read", flush=True)
pages = pd.concat(parts).drop_duplicates("id"); ended = pages.finish == "stop"
by = pages.groupby("scan")
scans = pd.DataFrame({
    "pages_read": by.finish.count().astype("int32"),
    "read_disagreement": (pages[ended].groupby("scan").edits.sum() / pages[ended].groupby("scan").label_chars.sum().clip(lower=1)).astype("float32"),
    "read_looped": pages.assign(looped=pages.finish.notna() & ~ended).groupby("scan").looped.any(),
    "turned": pages.assign(turned=(pages.turn != 0) & (pages.p > 0.9) & pages.turn.notna()).groupby("scan").turned.any(),
}).reset_index()
scans["read_looped"] = scans.read_looped.astype(object).where(scans.pages_read > 0, None)  # no page read: not known
image_of = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "scan"], filter="pool != 'gt'").to_pandas().set_index("scan").image
scans["image"] = scans.scan.map(image_of); scans = scans[scans.image.notna()]
features = pa.table({"image": pa.array(scans.image, pa.utf8()), "pages_read": pa.array(scans.pages_read, pa.int32()), "read_disagreement": pa.array(scans.read_disagreement, pa.float32(), from_pandas=True),
                     "read_looped": pa.array(scans.read_looped, pa.bool_(), from_pandas=True), "turned": pa.array(scans.turned.astype(bool), pa.bool_())})
print(f"{add_features(HOME / 'scans.lance', features):,} scans given the page checks;", "read:", int((scans.pages_read > 0).sum()), "| looped:", int(scans.read_looped.fillna(False).sum()), "| turned:", int(scans.turned.sum()),
      "| disagreement median %.3f" % scans.read_disagreement.median())
