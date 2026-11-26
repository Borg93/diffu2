"""Jev's scores of the scans it has scored, as columns of scans.lance (CURATION E62, column F1).

    flock runs/htr/curate/scans.lock uv run python runs/htr/curate/prototypes/jev_to_lance.py

``runs/htr/curate/jev/scores/part-*.jsonl`` holds a row a scan, keyed by the scan's ``image``: Jev's score of each
line of the old label against its line image, gathered as ``jev_page`` (the lines' mean), ``jev_low`` (the share of
lines scored low), ``jev_worst`` (the lowest line) and ``jev_lines`` (how many lines were scored). Run again as more
scans are scored: it reads the files as they stand.
"""

import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import json
from pathlib import Path

import lance
import pyarrow as pa

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate")


def main() -> None:
    rows: dict[str, dict] = {}
    for path in sorted((HOME / "jev/scores").glob("part-*.jsonl")):
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line:
                row = json.loads(line)
                rows[row["image"]] = row
    scored = list(rows.values())
    features = pa.table(
        {
            "image": pa.array([row["image"] for row in scored], pa.utf8()),
            "jev_page": pa.array([row["mean"] for row in scored], pa.float32()),
            "jev_low": pa.array([row["low"] for row in scored], pa.float32()),
            "jev_worst": pa.array([row["worst"] for row in scored], pa.float32()),
            "jev_lines": pa.array([row["lines"] for row in scored], pa.int32()),
        }
    )
    table = lance.dataset(str(HOME / "scans.lance"))
    before = table.version
    matched = add_features(HOME / "scans.lance", features)
    table = lance.dataset(str(HOME / "scans.lance"))
    print(f"{matched:,} of {len(scored):,} scored scans given Jev's columns; version {before} -> {table.version}")
    print(table.to_table(columns=features.column_names, filter="jev_page IS NOT NULL", limit=5).to_pandas().to_string())


if __name__ == "__main__":
    main()
