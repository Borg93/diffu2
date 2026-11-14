"""The calibration pages of the Svea pretraining pool: scans that are in GT-Lejonet and whose ALTO in the pool is still
the old pipeline's (htrflow: YOLO regions and lines, TrOCR), so the old label and the transcribers' truth lie side by
side. For each, the old lines are matched to the truth's by their boxes, and the old label's error is measured: per
line its CER against the truth's line, per page the lines missed and the lines astray, the polygons' cover of each
other, and the page's CER over its matched lines. Writes lines.parquet, pages.parquet and sources.tsv (every GT scan in
the pool with its label's source) to --out. --pool is the folder of the pool's volumes of ALTO files (the Svea pool
by default).

    uv run python packages/htr_train/lion_data_prep/curate/calibration.py --out runs/htr/curate/calibration
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from diffu_htr_train.sets.jev_pairs import edit_distance

NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
POOL = Path("/mnt/work/haas/deliveries_old/svea-all")
GT = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
METAS = [Path(f"data/htr/lightonocr3_pages.{split}.meta.jsonl") for split in ("train", "dev", "test")]
SCALE = 8  # masks are drawn at 1/8 of the scan
MATCH_IOU = 0.2  # an old line and a true line are the same line from this overlap of their boxes


class Line(BaseModel):
    """A line of an ALTO page: its box (x0, y0, x1, y1), its outline (the box's corners when it has none), its text."""

    model_config = ConfigDict(frozen=True)

    box: tuple[float, float, float, float]
    outline: list[tuple[float, float]]
    text: str


def source(path: Path) -> str:
    """Which tool wrote an ALTO file, by its description."""
    head = path.read_text(encoding="utf-8")[:3000]
    return "htrflow" if "htrflow" in head else "transkribus" if "Transkribus" in head else "other"


def points(text: str) -> list[tuple[float, float]]:
    numbers = [float(n) for n in text.replace(",", " ").split()]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def read_alto(path: Path) -> tuple[tuple[int, int], list[Line]]:
    """An ALTO page's size and its lines that hold text."""
    root = ET.parse(path).getroot()
    page = root.find(".//a:Page", NS)
    if page is None:
        raise ValueError(f"{path}: no Page")
    lines = []
    for element in page.iterfind(".//a:TextLine", NS):
        text = " ".join(s.get("CONTENT", "") for s in element.iterfind("a:String", NS)).strip()
        if not text:
            continue
        x, y = float(element.get("HPOS", 0)), float(element.get("VPOS", 0))
        box = (x, y, x + float(element.get("WIDTH", 0)), y + float(element.get("HEIGHT", 0)))
        polygon = element.find("a:Shape/a:Polygon", NS)
        outline = points(polygon.get("POINTS", "")) if polygon is not None else []
        if len(outline) < 3:
            outline = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
        lines.append(Line(box=box, outline=outline, text=text))
    return (int(float(page.get("WIDTH", 0))), int(float(page.get("HEIGHT", 0)))), lines


def mask(lines: list[Line], size: tuple[int, int]) -> np.ndarray:
    image = Image.new("1", (size[0] // SCALE + 1, size[1] // SCALE + 1))
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.polygon([(x / SCALE, y / SCALE) for x, y in line.outline], fill=1)
    return np.asarray(image, dtype=bool)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    inter = width * height
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(old: list[Line], true: list[Line]) -> list[tuple[int, int, float]]:
    """Old and true lines paired one to one, the most overlapping pairs first."""
    pairs = sorted(((box_iou(o.box, t.box), i, j) for i, o in enumerate(old) for j, t in enumerate(true)), reverse=True)
    used_old: set[int] = set()
    used_true: set[int] = set()
    matched = []
    for iou, i, j in pairs:
        if iou < MATCH_IOU:
            break
        if i in used_old or j in used_true:
            continue
        used_old.add(i)
        used_true.add(j)
        matched.append((i, j, iou))
    return matched


def gt_scans(pool: Path) -> dict[str, dict[str, str]]:
    """Every GT scan whose volume is in the pool: its split and collection, by scan stem."""
    volumes = {p.name for p in pool.iterdir()}
    scans: dict[str, dict[str, str]] = {}
    for meta in METAS:
        for row in map(json.loads, meta.open(encoding="utf-8")):
            stem = row["scan"].rsplit(".", 1)[0]
            if stem.split("_")[0] in volumes:
                scans.setdefault(stem, {"split": row["split"], "collection": row["collection"]})
    return scans


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    scans = gt_scans(args.pool)
    truths = {p.stem: p for p in GT.glob("*/*/*/alto/*.xml") if p.stem in scans}
    line_rows, page_rows, source_rows = [], [], []
    for stem, info in sorted(scans.items()):
        old_path = args.pool / stem.split("_")[0] / f"{stem}.xml"
        if not old_path.exists():
            continue
        wrote = source(old_path)
        source_rows.append({"scan": stem, **info, "source": wrote})
        if wrote != "htrflow" or stem not in truths:
            continue
        size, old = read_alto(old_path)
        _, true = read_alto(truths[stem])
        pairs = match(old, true)
        errors = chars = 0
        for i, j, iou in pairs:
            distance = edit_distance(old[i].text, true[j].text)
            errors += distance
            chars += len(true[j].text)
            width, height = old[i].box[2] - old[i].box[0], old[i].box[3] - old[i].box[1]
            line_rows.append({
                "scan": stem, "old": old[i].text, "truth": true[j].text, "iou": iou, "errors": distance,
                "cer": distance / max(1, len(true[j].text)), "width": width, "height": height,
                "chars_per_100px": 100 * len(old[i].text) / max(1.0, width),
            })  # fmt: skip
        old_mask, true_mask = mask(old, size), mask(true, size)
        inter = np.logical_and(old_mask, true_mask).sum()
        page_rows.append({
            "scan": stem, **info, "old_lines": len(old), "true_lines": len(true), "matched": len(pairs),
            "missed": len(true) - len(pairs), "astray": len(old) - len(pairs),
            "cer_matched": errors / max(1, chars),
            "old_on_true": inter / max(1, old_mask.sum()), "true_covered": inter / max(1, true_mask.sum()),
            "mask_iou": inter / max(1, np.logical_or(old_mask, true_mask).sum()),
        })  # fmt: skip
    pd.DataFrame(line_rows).to_parquet(args.out / "lines.parquet")
    pages = pd.DataFrame(page_rows)
    pages.to_parquet(args.out / "pages.parquet")
    sources = pd.DataFrame(source_rows)
    sources.to_csv(args.out / "sources.tsv", sep="\t", index=False)
    print(sources.groupby(["source", "split"]).size().to_string())
    print(pages.drop(columns=["scan", "split", "collection"]).describe().round(3).T.to_string())


if __name__ == "__main__":
    main()
