"""A random sample of the Svea pretraining pool's scans, staged as diffu-reader-set reads a GT: per volume an alto/
folder of links to the pool's ALTO files and, beside it, links to their scans. Volumes that hold any GT-Lejonet scan
are left out whole (their hands are in the fine-tuning and test sets), as are files the old pipeline (htrflow) did not
write and scans without an image. The sample is uniform over the pool's remaining scans, by --seed.

    uv run python packages/htr_train/lion_data_prep/curate/pilot_sample.py --scans 28000 --out /mnt/scratch/gabriel/htr/svea_pilot
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ALTO = Path("/mnt/work/haas/deliveries_old/svea-all")
SCANS = Path("/mnt/scratch/haas/input/svea")
METAS = [Path(f"data/htr/lightonocr3_pages.{split}.meta.jsonl") for split in ("train", "dev", "test")]


def gt_volumes() -> set[str]:
    """The pool's volumes that hold a GT-Lejonet scan."""
    volumes = {p.name for p in ALTO.iterdir()}
    return {
        volume
        for meta in METAS
        for row in map(json.loads, meta.open(encoding="utf-8"))
        if (volume := row["scan"].split("_")[0]) in volumes
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scans", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    left_out = gt_volumes()
    files = sorted(
        path for volume in sorted(ALTO.iterdir()) if volume.name not in left_out for path in volume.glob("*.xml")
    )
    random.Random(args.seed).shuffle(files)
    kept = 0
    for path in files:
        if kept == args.scans:
            break
        scan = SCANS / path.parent.name / f"{path.stem}.jpg"
        if "htrflow" not in path.read_text(encoding="utf-8")[:3000] or not scan.exists():
            continue
        folder = args.out / path.parent.name
        (folder / "alto").mkdir(parents=True, exist_ok=True)
        (folder / "alto" / path.name).symlink_to(path)
        (folder / scan.name).symlink_to(scan)
        kept += 1
    print(
        f"{kept} scans of {len(files)} in {len({p.parent for p in files})} volumes ({len(left_out)} GT volumes left out)"
    )


if __name__ == "__main__":
    main()
