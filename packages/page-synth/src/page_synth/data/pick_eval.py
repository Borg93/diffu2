"""Pick a fixed evaluation set from a split of page_synth.data.build_crops: one crop per collection, spread over them.

Each collection contributes its first crop with a margin note, heading or page number (else its first crop),
and when there are more collections than ``-n`` the picks are spread evenly over the alphabetical list, so
every archive group is represented. The picked images are hard-linked next to their metadata.

    python -m page_synth.data.pick_eval data_flux3/layout_crops_v2/val \
        data_flux3/layout_crops_v2/val_eval -n 24
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from page_synth.records import CropRecord, read_records


def has_role(record: CropRecord) -> bool:
    return any(line.role for line in record.lines)


def pick(records: list[CropRecord], count: int) -> list[CropRecord]:
    """One record per collection (preferring one with a role line), at most ``count``, spread over them."""
    chosen: dict[str, CropRecord] = {}
    for record in records:
        current = chosen.get(record.collection)
        if current is None or (has_role(record) and not has_role(current)):
            chosen[record.collection] = record
    names = sorted(chosen)
    if len(names) > count:
        names = list(dict.fromkeys(names[round(i * (len(names) - 1) / (count - 1))] for i in range(count)))
    return [chosen[name] for name in names]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", type=Path, help="a split folder: metadata.jsonl and the crops")
    ap.add_argument("out", type=Path)
    ap.add_argument("-n", type=int, default=24, help="crops to pick")
    args = ap.parse_args()

    records = read_records(args.split / "metadata.jsonl")
    picked = pick(records, args.n)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as out:
        for record in picked:
            image = args.out / record.file_name
            if not image.exists():
                os.link(args.split / record.file_name, image)
            out.write(f"{record.model_dump_json()}\n")
    print(f"{len(picked)} crops, {sum(has_role(r) for r in picked)} with a margin note, heading or page number")


if __name__ == "__main__":
    main()
