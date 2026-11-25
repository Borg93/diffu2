"""Every served reading of a pool set checked for a loop that ended by itself (CURATION E72, the reading's quality).

The pool read's server stops a reading that repeats (``finish`` ``repetition``); a reading can still loop on a
shorter span and end by itself ("Eric Eric Eric Eric"). This runs the RL reward's loop detector (NVIDIA's
``completion_quality``, 6-word n-grams four times, against the old label's own counts so a legitimate repeat is
not a loop) over every reading of the sets named, on the CPU in parallel, and writes per page
``runs/htr/curate/pool_read/loops/SET.parquet``: ``id``, ``finish``, ``looped``, ``max_repeated_ngram``,
``read_chars``, ``label_chars``.

    PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python \\
        runs/htr/curate/prototypes/loop_scan.py WORKERS SET [SET ...]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nemotron.recipes.parse.document_rewards import completion_quality

SETS = Path("/mnt/scratch/gabriel/htr")
READS = Path("runs/htr/curate/pool_read")
LOOP = {"ngram_size": 6, "min_repeats": 4, "token_run_limit": 32}


def lines(path: Path) -> list[str]:
    """The file's lines, split at newlines alone (``str.splitlines`` also splits at U+0085 inside a label)."""
    return [line for line in path.read_text(encoding="utf-8").split("\n") if line]


def check(item: tuple[str, str, str, str]) -> dict:
    page, finish, raw, label = item
    done = completion_quality(raw, reference_text=label, **LOOP)
    return {
        "id": page,
        "finish": finish,
        "looped": bool(done["looped"]),
        "max_repeated_ngram": int(done["max_repeated_ngram"]),
        "read_chars": len(raw),
        "label_chars": len(label),
    }


def main() -> None:
    workers, names = int(sys.argv[1]), sys.argv[2:]
    out_dir = READS / "loops"
    out_dir.mkdir(exist_ok=True)
    for name in names:
        rows = lines(SETS / f"{name}_nemotron/pages.jsonl")
        metas = [json.loads(line)["id"] for line in lines(SETS / f"{name}_nemotron/pages.meta.jsonl")]
        if len(rows) != len(metas):
            raise SystemExit(f"{name}: {len(rows)} pages, {len(metas)} lines of meta")
        label = {page: json.loads(row)["messages"][1]["content"] for page, row in zip(metas, rows, strict=True)}
        items = []
        for line in lines(READS / f"{name}_vllm.jsonl"):
            read = json.loads(line)
            items.append((read["id"], read.get("finish") or "", read["raw"], label[read["id"]]))
        with ProcessPoolExecutor(workers) as pool:
            found = list(pool.map(check, items, chunksize=256))
        pq.write_table(pa.Table.from_pylist(found), out_dir / f"{name}.parquet")
        stopped = [row for row in found if row["finish"] == "stop"]
        print(
            f"{name}: {len(found)} readings; ended by themselves {len(stopped)}, of them looping "
            f"{sum(row['looped'] for row in stopped)} ({100 * sum(row['looped'] for row in stopped) / max(1, len(stopped)):.2f} %)",
            flush=True,
        )


if __name__ == "__main__":
    main()
