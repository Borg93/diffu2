"""Our ground-truth page sets as a dataset bundle in the Parse recipe's manifest contract.

Writes ``train.jsonl``, ``monitor.jsonl``, ``test.jsonl``, ``images/`` (symbolic links to our page images, which
``dataset_identity`` follows and hashes) and ``artifact.json`` (the recipe's own ``dataset_identity``/``artifact``)::

    PYTHONPATH=/mnt/scratch/gabriel/src/Nemotron/src /mnt/scratch/gabriel/envs/parse_rlvr/venv/bin/python \\
        runs/htr/rlvr/build_dataset.py --out /mnt/scratch/gabriel/htr/rlvr/data

The splits, and where they differ from the recipe's:

* ``monitor``: the dev draw our checkpoints are scored on (at most 30 pages a dev volume, seed 0; 115 pages).
* ``test``: the scored test pages (the ids of ``--scored``), without those from a monitor volume: the recipe wants
  the splits' context ids (volumes) disjoint, and a page cannot be given another volume than its own. The pages left
  out are listed in ``left_out.json``.
* ``train``: the ground-truth TRAIN pages of every volume that has no page in monitor or test. These are the pages
  RLVR may draw its candidates from (the recipe mines fresh pages; a reward on handwriting needs a true text). The
  reader was trained on all 49,491 train pages, most of them from the test pages' volumes; that cannot be undone
  here, so the test split measures unseen pages of seen volumes, and RLVR adds no page of those volumes.

A row: ``id``, ``image`` (relative), ``prompt``, ``target``, ``context_id`` (collection/volume), ``split``,
``collection``, ``volume``, ``scan``, ``side``, and three measured fields the task filters on: ``visual_tokens``
(the image prefix, (h//16)*(w//64)+1 of the processor's padded size), ``target_tokens`` (with EOS), ``has_table``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[3]
SETS = REPO / "data/htr"
SCORED = REPO / "runs/htr/nemotron_parse/test_read_selftrain2_norepeat10/pages.jsonl"
MODEL = REPO / "runs/htr/nemotron-parse-selftrain-finetune2/output/checkpoint-3000"
CANVAS = (2048, 1664)  # the processor's final_size, height and width


def visual_tokens(height: int, width: int) -> int:
    """The image prefix's length, as ``NemotronParseImageProcessor`` sizes a page (fit the canvas, pad to the grid)."""
    ratio = width / height
    new_height, new_width = height, width
    if height > CANVAS[0]:
        new_height = CANVAS[0]
        new_width = int(new_height * ratio)
    if new_width > CANVAS[1]:
        new_width = CANVAS[1]
        new_height = int(new_width / ratio)
    return ((new_height + 15) // 16) * ((new_width + 63) // 64) + 1


def page_set(split: str) -> list[dict]:
    """A page set's rows with their meta, in the manifest's fields (image still absolute)."""
    samples = (SETS / f"nemotron_parse_pages.{split}.jsonl").read_text("utf-8").splitlines()
    metas = (SETS / f"nemotron_parse_pages.{split}.meta.jsonl").read_text("utf-8").splitlines()
    rows = []
    for line, meta_line in zip(samples, metas, strict=True):
        sample, meta = json.loads(line), json.loads(meta_line)
        rows.append(
            {
                "id": meta["id"],
                "image": sample["images"][0],
                "prompt": sample["messages"][0]["content"].replace("<image>", ""),
                "target": sample["messages"][1]["content"],
                "context_id": f"{meta['collection']}/{meta['volume']}",
                "collection": meta["collection"],
                "volume": meta["volume"],
                "scan": meta["scan"],
                "side": meta["side"],
            }
        )
    return rows


def drawn(pages: list[dict], per_volume: int, seed: int) -> list[dict]:
    """The dev draw of ``diffu_htr_train.nemotron_parse.checkpoint_cer.drawn`` (by volume name, as it does)."""
    by_volume: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_volume[page["volume"]].append(page)
    picked: list[dict] = []
    for volume in sorted(by_volume):
        held = sorted(by_volume[volume], key=lambda page: page["id"])
        picked.extend(held if len(held) <= per_volume else random.Random(seed).sample(held, per_volume))
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scored", type=Path, default=SCORED)
    parser.add_argument("--per-volume", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from nemotron.recipes.parse.common import artifact, dataset_identity
    from transformers import AutoTokenizer

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"A dataset is not changed once written; choose a new, empty directory: {out}")

    monitor = drawn(page_set("dev"), args.per_volume, args.seed)
    scored = {json.loads(line)["id"] for line in args.scored.read_text("utf-8").splitlines()}
    test_all = [row for row in page_set("test") if row["id"] in scored]
    if len(test_all) != len(scored):
        raise ValueError("A scored test page is not in the test page set")
    monitor_contexts = {row["context_id"] for row in page_set("dev")}  # the whole dev volumes, not only the draw
    test = [row for row in test_all if row["context_id"] not in monitor_contexts]
    test_contexts = {row["context_id"] for row in test}
    train_all = page_set("train")
    seen = Counter(row["context_id"] for row in train_all)
    train = [row for row in train_all if row["context_id"] not in test_contexts | monitor_contexts]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    (out / "images").mkdir()
    splits = {"train": train, "monitor": monitor, "test": test}
    for split, rows in splits.items():
        for row in rows:
            source = Path(row["image"])
            link = out / "images" / f"{row['id']}{source.suffix.lower()}"
            os.symlink(source, link)
            with Image.open(source) as image:
                width, height = image.size
            row.update(
                image=str(link.relative_to(out)),
                split=split,
                visual_tokens=visual_tokens(height, width),
                target_tokens=len(tokenizer.encode(row["target"], add_special_tokens=False)) + 1,
                has_table="\\begin{tabular}" in row["target"],
            )
        (out / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    clash = sorted({row["context_id"] for row in test_all} & set(seen))
    left_out = {
        "test_pages_scored": len(test_all),
        "test_pages_from_volumes_in_sft_training": sum(row["context_id"] in seen for row in test_all),
        "test_volumes_scored": len({row["context_id"] for row in test_all}),
        "test_volumes_in_sft_training": len(clash),
        "test_pages_left_out_as_monitor_volumes": sorted(
            row["id"] for row in test_all if row["context_id"] in monitor_contexts
        ),
        "sft_train_pages": len(train_all),
        "sft_train_volumes": len(seen),
        "train_pages_kept_for_rl": len(train),
        "train_volumes_kept_for_rl": len({row["context_id"] for row in train}),
        "train_pages_left_out_as_test_volumes": sum(seen[c] for c in clash),
    }
    (out / "left_out.json").write_text(json.dumps(left_out, indent=2, ensure_ascii=False) + "\n")
    identity = dataset_identity(out)
    artifact(out, "parse/dataset", out, identity=identity)
    print(json.dumps({**left_out, "pages": identity["pages"], "contexts": identity["contexts"]}, indent=2))


if __name__ == "__main__":
    main()
