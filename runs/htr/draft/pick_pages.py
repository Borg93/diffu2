"""Pick the pages a draft is trained and validated on from a pool the reader has read: spread over volumes.

Reads the reader's served answers (``id``, ``raw``, ``finish``; a file that may be growing, so only whole lines are
taken) and the pool's page set (``pages.jsonl`` with ``images[0]``, ``pages.meta.jsonl`` with ``id`` and ``volume``).
Of the pages that stopped by themselves with a non-empty answer, it draws by a fixed seed:

- validation: ``--val`` pages from ``--val-volumes`` volumes, none of which gives a training page;
- training: ``--train`` pages from the other volumes, at most ``cap`` a volume, ``cap`` being the smallest that
  reaches ``--train`` (so every volume gives pages and none dominates).

The answers file is taken as ``runs/htr/verify/parse_vllm_pool_read.py`` wrote it: that client splits the page set
with ``str.splitlines()``, which also splits at U+0085 inside a label, so after such a line the answer of a line's
image is filed under the next line's id (pool_v3: one U+0085 at line 203,141, every later page moved by one). The
image each answer belongs to is worked out the same way here; ``filed_as`` in the answers written is the id it was
found under.

Volumes of ``--exclude`` page sets (the benchmark) give nothing. Writes, for each split, ``NAME.jsonl`` +
``NAME.meta.jsonl`` (the page set ``make_cache.py --pages`` takes) and ``NAME.answers.jsonl`` (its ``--answers``).

    python3 runs/htr/draft/pick_pages.py --out /mnt/scratch/gabriel/htr/draft/sets/v1
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    """Write the two splits."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=Path, default=Path("runs/htr/curate/pool_read/pool_v3_vllm.jsonl"))
    parser.add_argument("--pages", type=Path, default=Path("/mnt/scratch/gabriel/htr/pool_v3_nemotron/pages.jsonl"))
    parser.add_argument(
        "--exclude", type=Path, nargs="*", default=[Path("/mnt/scratch/gabriel/htr/parse_bench/pages.meta.jsonl")]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=4000)
    parser.add_argument("--val", type=int, default=200)
    parser.add_argument("--val-volumes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    answers: dict[str, dict] = {}
    with args.answers.open("rb") as lines:
        for line in lines:
            if not line.endswith(b"\n"):
                break  # a line still being written
            row = json.loads(line)
            if row.get("finish") == "stop" and row.get("raw", "").strip():
                answers[row["id"]] = {"id": row["id"], "raw": row["raw"], "finish": "stop", "tokens": row.get("tokens")}

    barred = {json.loads(line)["volume"] for path in args.exclude for line in path.read_text("utf-8").splitlines()}
    meta_path = args.pages.with_name(args.pages.name.replace(".jsonl", ".meta.jsonl"))
    with meta_path.open(encoding="utf-8") as metas:
        ids = [json.loads(line)["id"] for line in metas]
    by_volume: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen = moved = 0  # lines as the reading client counted them
    with args.pages.open(encoding="utf-8", newline="\n") as rows, meta_path.open(encoding="utf-8") as metas:
        for row, meta_line in zip(rows, metas, strict=True):
            # the client split the page set with str.splitlines(), which also splits at U+0085 and its like inside a
            # label: from such a line on, the answer of this line's image is filed under a later line's id
            pieces = row.rstrip("\n").splitlines()
            filed_as = ids[seen] if len(pieces) == 1 and seen < len(ids) else None
            seen += len(pieces)
            meta = json.loads(meta_line)
            if filed_as in answers and meta["volume"] not in barred:
                moved += filed_as != meta["id"]
                by_volume[meta["volume"]].append((meta["id"], json.loads(row)["images"][0], filed_as))
    print(f"{moved} of {sum(map(len, by_volume.values()))} usable answers are filed under another page's id")

    volumes = sorted(by_volume)
    rng.shuffle(volumes)
    val_volumes, train_volumes = volumes[: args.val_volumes], volumes[args.val_volumes :]

    def drawn(names: list[str], wanted: int) -> list[tuple[str, str, str]]:
        cap = 1
        while sum(min(cap, len(by_volume[name])) for name in names) < wanted and cap < 10**6:
            cap += 1
        picked = [
            (page_id, image, name, filed_as)
            for name in names
            for page_id, image, filed_as in rng.sample(sorted(by_volume[name]), min(cap, len(by_volume[name])))
        ]
        rng.shuffle(picked)
        print(f"  cap {cap} a volume, {len({name for _, _, name, _ in picked[:wanted]})} volumes")
        return picked[:wanted]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(answers)} answers that stopped, {len(volumes)} volumes with one ({len(barred)} volumes barred)")
    for name, names, wanted in (("train", train_volumes, args.train), ("val", val_volumes, args.val)):
        print(name)
        picked = drawn(names, wanted)
        (args.out / f"{name}.jsonl").write_text(
            "".join(json.dumps({"images": [image]}) + "\n" for _, image, _, _ in picked), "utf-8"
        )
        (args.out / f"{name}.meta.jsonl").write_text(
            "".join(json.dumps({"id": page_id, "volume": volume}) + "\n" for page_id, _, volume, _ in picked), "utf-8"
        )
        (args.out / f"{name}.answers.jsonl").write_text(
            "".join(
                json.dumps({**answers[filed_as], "id": page_id, "filed_as": filed_as}, ensure_ascii=False) + "\n"
                for page_id, _, _, filed_as in picked
            ), "utf-8"
        )
        print(f"  {len(picked)} pages in {args.out / name}.jsonl")


if __name__ == "__main__":
    main()
