"""The Space's example regions, cut from the GT-Lejonet sets: each region's masked crop, its lines' crops and its
lines' outlines (from the GT's PAGE XML, as fractions of the crop), in OUT with an examples.json.

    python build_examples.py --sets data/htr --gt /mnt/work/GT-Lejonet_HTR_202602 --out examples

The regions are dev regions (volumes held out from training) whose lines are all in the dev lines with the same
text. A region's crop is the box of its outline and its lines' outlines together (diffu_page's reader_set).
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image

REGIONS = (  # one or two per collection, short to long
    "8d24e7df7dea_2",
    "19a1b2195bab_right_0",
    "eed569a6699c_left_3",
    "f970d28ac08b_left_0",
    "38c4ce6331ad_left_1",
    "35ffe6b8b600_left_0",
)
MASKED = "minicpm_regions_masked_crops"  # the regions as the reader was trained on them, beside the plain crops


def rows(sets: Path, stem: str) -> dict[str, dict[str, Any]]:
    """A set's samples by id: its image, its text and its meta."""
    samples = (sets / f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()
    metas = (sets / f"{stem}.meta.jsonl").read_text(encoding="utf-8").splitlines()
    found = {}
    for sample, meta in zip(samples, metas, strict=True):
        sample, meta = json.loads(sample), json.loads(meta)
        found[meta["id"]] = meta | {"image": sample["images"][0], "text": sample["messages"][1]["content"]}
    return found


def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def outline(element: ET.Element) -> list[tuple[int, int]]:
    coords = next((child for child in element if local(child.tag) == "Coords"), None)
    if coords is None:
        return []
    return [(int(float(x)), int(float(y))) for x, y in (point.split(",") for point in coords.get("points", "").split())]


def transcription(line: ET.Element) -> str:
    for child in line:
        if local(child.tag) == "TextEquiv":
            return " ".join((child.findtext("{*}Unicode") or "").split())
    return ""


def line_outlines(page: Path, texts: list[str]) -> list[list[tuple[float, float]]]:
    """The outlines of the region of ``page`` whose lines read ``texts``, as fractions of the region's crop."""
    wanted = [" ".join(text.split()) for text in texts]
    for region in ET.parse(page).getroot().iter():
        if local(region.tag) != "TextRegion":
            continue
        lines = [child for child in region if local(child.tag) == "TextLine"]
        if [transcription(line) for line in lines] != wanted:
            continue
        shapes = [outline(line) for line in lines]
        points = outline(region) + [point for shape in shapes if len(shape) >= 3 for point in shape]
        left, top = min(x for x, _ in points), min(y for _, y in points)
        width, height = max(x for x, _ in points) - left, max(y for _, y in points) - top
        return [[(round((x - left) / width, 4), round((y - top) / height, 4)) for x, y in shape] for shape in shapes]
    raise ValueError(f"{page}: no region with these {len(texts)} lines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", type=Path, required=True, help="the folder of the dev sets (data/htr)")
    parser.add_argument("--gt", type=Path, required=True, help="GT-Lejonet's folder, for the PAGE XML")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lines, regions = rows(args.sets, "minicpm_lines_scaled.dev"), rows(args.sets, "minicpm_regions.dev")
    examples = []
    for region_id in REGIONS:
        region = regions[region_id]
        own = sorted(
            (line for line_id, line in lines.items() if line_id.rsplit("_", 1)[0] == region_id),
            key=lambda line: int(line["id"].rsplit("_", 1)[1]),
        )
        texts = [line["text"] for line in own]
        if [text.strip() for text in texts] != [text.strip() for text in region["text"].split("\n")]:
            raise ValueError(f"{region_id}: its dev lines are not its region's lines")
        folder = args.out / region_id
        folder.mkdir(parents=True, exist_ok=True)
        masked = Path(region["image"]).parent.parent / MASKED / Path(region["image"]).name
        shutil.copy(masked, folder / "region.png")
        for i, line in enumerate(own):
            shutil.copy(line["image"], folder / f"line_{i:02d}.jpg")
        found = list((args.gt / region["collection"]).rglob(f"page/{Path(region['scan']).stem}.xml"))
        if len(found) != 1:
            raise ValueError(f"{region_id}: {len(found)} PAGE files for {region['scan']}")
        with Image.open(masked) as image:
            size = list(image.size)
        examples.append(
            {
                "id": region_id,
                "collection": region["collection"],
                "volume": region["volume"],
                "scan": region["scan"],
                "lines": texts,
                "region_size": size,
                "outlines": line_outlines(found[0], texts),
            }
        )
    (args.out / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(examples)} regions, {sum(len(example['lines']) for example in examples)} lines in {args.out}")


if __name__ == "__main__":
    main()
