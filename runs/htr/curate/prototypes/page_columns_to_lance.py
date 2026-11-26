"""Per-page columns of train_pages.lance filled from what stands on disk, joined on ``page`` (CURATION, columns F2 to F5).

    flock runs/htr/curate/pages.lock nice -n 10 uv run --with rapidfuzz python \\
        runs/htr/curate/prototypes/page_columns_to_lance.py STEP [SET ...]

``STEP`` is one of:

``reads``    the reader's text of the sets named (``pool_read/SET_vllm.jsonl``): ``read_nemotron``, ``read_finish``,
             ``read_chars``, ``read_disagreement``, computed as ``pages_table.py`` computes them (NFC, ends trimmed;
             the edits between the label and the reading, both with line breaks and runs of spaces as one space, over
             the label's characters). Run again as a read goes on: pages read since take their values.
``rereads``  the pages the server's repetition stop cut, read again by the guarded reader on the CPU
             (``pool_read/reread_cpu/SET.shard*.jsonl``): ``reread_nemotron``, ``reread_finish``,
             ``reread_disagreement``. The served reading keeps its columns.
``loops``    the loop detector's verdict on the served reading (``pool_read/loops/SET.parquet``, by ``loop_scan.py``):
             ``read_looped``, ``max_repeated_ngram``.
``scans``    the page's scan's columns of scans.lance copied to the page, on ``image``: SCAN_COLUMNS.

Every step tags the table's version before it writes (``before-STEP-DATE``), prints five pages before and after, and
is a join (``add_features``): no row moves, pages not named keep what they hold. Nothing else may write the table
meanwhile.
"""

import os

os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
import contextlib
import json
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
from rapidfuzz.distance import Levenshtein

from diffu_curate.table import add_features

HOME = Path("runs/htr/curate")
PAGES = HOME / "train_pages.lance"
READS = HOME / "pool_read"
SCAN_COLUMNS = ("quality_bad", "quality_cer", "language", "kind", "is_table", "cut_class", "pool_cluster")
SAMPLE = 5
WORKERS = 16


def clean(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def flat(text: str) -> str:
    """Line breaks and runs of spaces as one space: the reading has no line breaks."""
    return " ".join(text.split())


def disagreement(pair: tuple[str, str]) -> float:
    label, read = flat(pair[0]), flat(pair[1])
    return Levenshtein.distance(label, read) / max(1, len(label))


def records(path: Path) -> list[dict]:
    """The file's rows, split at newlines alone (``str.splitlines`` also splits at U+0085 inside a label); a last
    line still being written is left out."""
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line]
    rows = [json.loads(line) for line in lines[:-1]]
    with contextlib.suppress(json.JSONDecodeError, IndexError):
        rows.append(json.loads(lines[-1]))
    return rows


def labels(pages: list[str], sets: list[str] | None = None) -> dict[str, str]:
    """The table's label of each of ``pages`` that it holds, in the sets named alone if any is (a page two sets
    hold is a row of the first, and keeps that set's reading)."""
    wanted, found = set(pages), {}
    where = None if sets is None else "set IN (" + ", ".join(f"'{name}'" for name in sets) + ")"
    table = lance.dataset(str(PAGES))
    for batch in table.to_batches(columns=["page", "label"], filter=where, batch_size=65536):
        for page, label in zip(batch.column("page").to_pylist(), batch.column("label").to_pylist(), strict=True):
            if page in wanted:
                found[page] = label
    return found


def disagreements(pairs: list[tuple[str, str]]) -> list[float]:
    with ProcessPoolExecutor(WORKERS) as pool:
        return list(pool.map(disagreement, pairs, chunksize=2000))


def reads(names: list[str]) -> pa.Table:
    rows: dict[str, tuple[str, str | None]] = {}
    for name in names:
        for row in records(READS / f"{name}_vllm.jsonl"):
            rows.setdefault(row["id"], (clean(row["prediction"]), row.get("finish")))
        print(name, "read;", len(rows), "pages so far", flush=True)
    label = labels(list(rows), names)
    pages = [page for page in rows if page in label]
    scores = disagreements([(label[page], rows[page][0]) for page in pages])
    return pa.table(
        {
            "page": pa.array(pages, pa.utf8()),
            "read_nemotron": pa.array([rows[page][0] for page in pages], pa.utf8()),
            "read_finish": pa.array([rows[page][1] for page in pages], pa.utf8()),
            "read_chars": pa.array([len(rows[page][0]) for page in pages], pa.int32()),
            "read_disagreement": pa.array(scores, pa.float32()),
        }
    )


def set_of_page() -> dict[str, str]:
    """The set whose row each page of the table is (a page two sets hold is a row of the first)."""
    held = lance.dataset(str(PAGES)).to_table(columns=["page", "set"])
    return dict(zip(held.column("page").to_pylist(), held.column("set").to_pylist(), strict=True))


def rereads() -> pa.Table:
    home = set_of_page()
    rows: dict[str, tuple[str, str]] = {}
    seen = 0
    for path in sorted((READS / "reread_cpu").glob("*.shard*.jsonl")):
        for row in records(path):
            seen += 1
            if home.get(row["id"]) == row["set"]:
                rows[row["id"]] = (clean(row["prediction"]), row["finish"])
    label = labels(list(rows))
    pages = list(rows)
    print(seen, "pages read again;", len(pages), "of them rows of their own set in the table", flush=True)
    return pa.table(
        {
            "page": pa.array(pages, pa.utf8()),
            "reread_nemotron": pa.array([rows[page][0] for page in pages], pa.utf8()),
            "reread_finish": pa.array([rows[page][1] for page in pages], pa.utf8()),
            "reread_disagreement": pa.array(
                disagreements([(label[page], rows[page][0]) for page in pages]), pa.float32()
            ),
        }
    )


def loops() -> pa.Table:
    home = set_of_page()
    parts = []
    for path in sorted((READS / "loops").glob("*.parquet")):
        part = pd.read_parquet(path)
        parts.append(part[part.id.map(home) == path.stem])
    found = pd.concat(parts)
    print(len(found), "readings with a loop verdict;", int(found.looped.sum()), "looped", flush=True)
    return pa.table(
        {
            "page": pa.array(found.id.astype(str), pa.utf8()),
            "read_looped": pa.array(found.looped.astype(bool), pa.bool_()),
            "max_repeated_ngram": pa.array(found.max_repeated_ngram, pa.int32()),
        }
    )


def scan_copies() -> pa.Table:
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", *SCAN_COLUMNS])
    pages = lance.dataset(str(PAGES)).to_table(columns=["page", "image"])
    joined = pages.join(scans, "image", join_type="inner")
    print(len(joined), "of", len(pages), "pages have their scan in scans.lance", flush=True)
    return joined.drop_columns(["image"])


def show(pages: list[str], columns: list[str], when: str) -> None:
    table = lance.dataset(str(PAGES))
    held = [column for column in columns if column in table.schema.names]
    quoted = ", ".join(f"'{page}'" for page in pages)
    frame = table.to_table(columns=["page", "set", *held], filter=f"page IN ({quoted})").to_pandas()
    for column in frame.columns:
        if frame[column].dtype == object or str(frame[column].dtype).startswith("str"):
            frame[column] = frame[column].str.slice(0, 60)
    print(f"-- {when}:\n{frame.to_string(index=False)}", flush=True)


def main() -> None:
    step, names = sys.argv[1], sys.argv[2:]
    makers = {"reads": lambda: reads(names), "rereads": rereads, "loops": loops, "scans": scan_copies}
    if step not in makers or (step == "reads") != bool(names):
        raise SystemExit(__doc__)
    features = makers[step]()
    columns = [name for name in features.column_names if name != "page"]
    table = lance.dataset(str(PAGES))
    step_ = max(1, len(features) // SAMPLE)
    sample = features.column("page").take(list(range(0, len(features), step_))[:SAMPLE]).to_pylist()
    before = table.version
    tag = f"before-{step}-{time.strftime('%d%b-%H%M').lower()}"
    table.tags.create(tag, before)
    show(sample, columns, f"before (version {before}, tag {tag})")
    matched = add_features(PAGES, features, key="page")
    table = lance.dataset(str(PAGES))
    show(sample, columns, f"after (version {table.version})")
    counts = {column: table.count_rows(f"{column} IS NOT NULL") for column in columns}
    print(f"{step}: {matched:,} pages given {columns}; version {before} -> {table.version}; with a value: {counts}")


if __name__ == "__main__":
    main()
