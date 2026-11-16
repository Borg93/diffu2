"""Our reader's readings of pool pages, joined to pages.lance as columns on the page id (CURATION E12, E30).

Each reading (pool_read/*.jsonl) names its page by the reader set's id; the set's meta file gives the scan and side,
which is the page table's ``page``. Columns: ``read_nemotron`` (the reading), ``read_label`` (the old label as the
reader set held it) and ``read_disagreement`` (their character error rate, the label as reference).

    uv run python runs/htr/curate/prototypes/reads_to_lance.py
"""
import json
from pathlib import Path

import pyarrow as pa
from rapidfuzz.distance import Levenshtein

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate"); SETS = Path("/mnt/scratch/gabriel/htr")

if __name__ == "__main__":
    rows: dict[str, tuple[str, str, float]] = {}
    for name in ("svea", "andra"):
        page_of = {}
        for line in (SETS / f"{name}_pilot_nemotron" / "pages.meta.jsonl").open(encoding="utf-8"):
            meta = json.loads(line); page_of[meta["id"]] = f"{Path(meta['scan']).stem}_{meta['side']}"
        for path in sorted((HOME / "pool_read").glob(f"{name}_pilot*.jsonl")):
            for line in path.open(encoding="utf-8"):
                read = json.loads(line); label, said = read["text"], read["prediction"]
                rows[page_of[read["id"]]] = (said, label, Levenshtein.distance(said, label) / max(1, len(label)))
    pages = sorted(rows)
    features = pa.table({"page": pages, "read_nemotron": [rows[page][0] for page in pages], "read_label": [rows[page][1] for page in pages],
                         "read_disagreement": pa.array([rows[page][2] for page in pages], pa.float32())})
    print(features.num_rows, "readings,", add_features(HOME / "pages.lance", features, key="page"), "matched")
