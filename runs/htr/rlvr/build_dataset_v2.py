"""Dataset bundle v2 for pilot 2: the same monitor and test pages as v1, RL candidates from every training volume
outside the dev volumes, and the context of a page is its scan (not its volume).

    PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src:runs/htr/rlvr \\
        /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python runs/htr/rlvr/build_dataset_v2.py \\
        --out /mnt/scratch/gabriel/htr/rlvr/data_v2

What changes from v1 (``build_dataset.py``, left as it is) and why:

* ``monitor`` = v1's 115 dev pages, ``test`` = v1's 197 test pages, by id and in v1's order (checked against the dev
  draw ``drawn(dev, 30, 0)`` again).
* ``train`` = every SFT training page whose volume is not one of the 5 dev volumes (v1: also none of the 77 volumes
  of the test pages, which left 9,811 pages of 144 volumes). The test pages' volumes were in SFT training already
  (page-disjoint), so RL on other pages of them is the same situation the test split measures. A training page that
  shares its scan with any dev or test page (by collection/volume/scan, and by the id's scan hash) is left out.
* ``context_id`` = ``collection/volume/scan`` in all three splits (v1: ``collection/volume``). The recipe's dataset
  check (``dataset_identity``: the splits' contexts are disjoint) then means "no scan in two splits"; that volumes
  are kept apart only for dev is asserted here and recorded in the artifact as a deviation.
* Manifests are written with ``ensure_ascii`` (escaped JSON): the recipe reads them with ``str.splitlines()``, which
  would split a row at a raw U+0085 or U+2028 inside a label. The source page sets are read line by line on "\\n".

Each row has v1's fields (``visual_tokens``, ``target_tokens``, ``has_table`` ...) and what the targeted candidate
draw reads (``htr_candidates``): ``lines``, ``line_median``, ``marked_lines``, ``leader_lines``, and from a 64-pixel
grey thumbnail ``brightness`` (mean grey level, 0..255) and ``dark_share`` (share of its pixels below 60).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from build_dataset import MODEL, drawn, visual_tokens
from htr_candidates import page_features
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SETS = REPO / "data/htr"
V1 = Path("/mnt/scratch/gabriel/htr/rlvr/data")
DEVIATION = (
    "context_id is collection/volume/scan in all three splits (the recipe's disjointness check holds per scan); "
    "train holds every SFT training page outside the 5 dev volumes, including pages of the 77 volumes the 197 test "
    "pages come from (page- and scan-disjoint from them; those volumes were in SFT training already). v1 used "
    "context_id = collection/volume and left the test volumes out of train."
)


def read_lines(path: Path) -> list[str]:
    """A JSON-lines file's lines, split on "\\n" only (``str.splitlines`` also splits at U+0085 inside a label)."""
    return [line for line in path.read_text("utf-8").split("\n") if line.strip()]


def page_set(split: str) -> list[dict]:
    """A page set's rows with their meta, in the manifest's fields (image still absolute)."""
    samples = read_lines(SETS / f"nemotron_parse_pages.{split}.jsonl")
    metas = read_lines(SETS / f"nemotron_parse_pages.{split}.meta.jsonl")
    rows = []
    for line, meta_line in zip(samples, metas, strict=True):
        sample, meta = json.loads(line), json.loads(meta_line)
        rows.append(
            {
                "id": meta["id"],
                "image": sample["images"][0],
                "prompt": sample["messages"][0]["content"].replace("<image>", ""),
                "target": sample["messages"][1]["content"],
                "context_id": f"{meta['collection']}/{meta['volume']}/{meta['scan']}",
                "collection": meta["collection"],
                "volume": meta["volume"],
                "scan": meta["scan"],
                "side": meta["side"],
            }
        )
    return rows


def volume(row: dict) -> tuple[str, str]:
    return row["collection"], row["volume"]


def scan_hash(row: dict) -> str:
    """The scan part of a page id (``5d98bb8b3fc6_right`` -> ``5d98bb8b3fc6``)."""
    return row["id"].split("_", 1)[0]


def measured(path: str) -> tuple[int, int, float, float]:
    """An image's width and height, and its 64-pixel grey thumbnail's mean level and share of pixels below 60."""
    with Image.open(path) as image:
        width, height = image.size
        image.draft("L", (128, 128))
        grey = image.convert("L")
    grey.thumbnail((64, 64))
    histogram = grey.histogram()
    pixels = sum(histogram)
    return (
        width,
        height,
        sum(level * count for level, count in enumerate(histogram)) / pixels,
        sum(histogram[:60]) / pixels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v1", type=Path, default=V1, help="the v1 bundle whose monitor and test pages are kept")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    from nemotron.recipes.parse.common import artifact, dataset_identity, validate_dataset
    from transformers import AutoTokenizer

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"A dataset is not changed once written; choose a new, empty directory: {out}")

    dev, test_all, train_all = page_set("dev"), page_set("test"), page_set("train")
    v1_monitor = [json.loads(line)["id"] for line in read_lines(args.v1 / "monitor.jsonl")]
    v1_test = [json.loads(line)["id"] for line in read_lines(args.v1 / "test.jsonl")]
    if [row["id"] for row in drawn(dev, 30, 0)] != v1_monitor:
        raise ValueError("The dev draw no longer gives v1's monitor pages")
    by_id = {row["id"]: row for row in dev + test_all}
    monitor = [by_id[page] for page in v1_monitor]
    test = [by_id[page] for page in v1_test]

    dev_volumes = {volume(row) for row in dev}
    held_scans = {row["context_id"] for row in dev + test_all}
    held_hashes = {scan_hash(row) for row in dev + test_all}
    counts = Counter()
    train = []
    for row in train_all:
        if volume(row) in dev_volumes:
            counts["dev_volume"] += 1
        elif row["context_id"] in held_scans or scan_hash(row) in held_hashes:
            counts["scan_of_a_dev_or_test_page"] += 1
        else:
            train.append(row)

    # The checks the recipe's context check no longer makes once contexts are scans.
    dev_names = {name for _, name in dev_volumes}
    assert not any(volume(row) in dev_volumes or row["volume"] in dev_names for row in train), "train page of a dev volume"
    assert not {row["context_id"] for row in train} & {row["context_id"] for row in monitor + test}, "shared scan"
    assert not {scan_hash(row) for row in train} & {scan_hash(row) for row in monitor + test}, "shared scan hash"
    assert not {row["id"] for row in train} & {row["id"] for row in monitor + test}, "shared page"
    assert not {volume(row) for row in test} & dev_volumes, "test page of a dev volume"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    splits = {"train": train, "monitor": monitor, "test": test}
    everything = [row for rows in splits.values() for row in rows]
    with Pool(args.workers) as pool:
        sizes = pool.map(measured, [row["image"] for row in everything], chunksize=64)
    tokens = tokenizer([row["target"] for row in everything], add_special_tokens=False)["input_ids"]
    (out / "images").mkdir()
    for row, (width, height, brightness, dark), ids in zip(everything, sizes, tokens, strict=True):
        source = Path(row["image"])
        link = out / "images" / f"{row['id']}{source.suffix.lower()}"
        os.symlink(source, link)
        row.update(
            image=str(link.relative_to(out)),
            visual_tokens=visual_tokens(height, width),
            target_tokens=len(ids) + 1,
            brightness=round(brightness, 2),
            dark_share=round(dark, 4),
            **page_features(row["target"]),
        )
    for split, rows in splits.items():
        for row in rows:
            row["split"] = split
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = {
        "deviation": DEVIATION,
        "v1": str(args.v1),
        "pages": {split: len(rows) for split, rows in splits.items()},
        "scans": {split: len({row["context_id"] for row in rows}) for split, rows in splits.items()},
        "volumes": {split: len({volume(row) for row in rows}) for split, rows in splits.items()},
        "sft_train_pages": len(train_all),
        "sft_train_volumes": len({volume(row) for row in train_all}),
        "sft_train_pages_left_out": dict(counts),
        "train_pages_of_test_volumes": sum(volume(row) in {volume(t) for t in test} for row in train),
        "train_volumes_shared_with_test": len({volume(row) for row in train} & {volume(t) for t in test}),
        "dev_volumes": sorted("/".join(pair) for pair in dev_volumes),
        "train_tables": sum(row["has_table"] for row in train),
        "train_brightness_below_100": sum(row["brightness"] < 100 for row in train),
        "checks": [
            "no train page of a dev volume (collection/volume, and volume name)",
            "no train scan (collection/volume/scan, and the id's scan hash) equal to a monitor or test scan",
            "no train page id in monitor or test",
            "contexts (scans) disjoint between all three splits (the recipe's dataset_identity)",
        ],
    }
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    identity = dataset_identity(out)
    artifact(
        out,
        "parse/dataset",
        out,
        identity=identity,
        built_by="runs/htr/rlvr/build_dataset_v2.py",
        context="collection/volume/scan",
        deviation=DEVIATION,
        build_report=report,
    )
    validate_dataset(out)  # the stage's own check
    print(json.dumps({**report, "identity_pages": identity["pages"], "contexts": identity["contexts"]}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
