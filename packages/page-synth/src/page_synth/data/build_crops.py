"""FLUX.3 Klein LoRA training crops with layout captions, from the HTR ground truth (PAGE XML pages).

Archival spreads are ~5000 px wide with 60-120 lines, so a whole page at FLUX's ~1 MP shrinks the letters
to a few pixels. Each crop instead covers consecutive WHOLE lines of one text column (a text region plus
the margin notes, headings and page numbers beside, above or below it), scaled so the line spacing is
about ``--line-spacing`` px and saved at the exact size of a NeMo bucket. Its caption follows BFL's
layout-prompt guide: the collection, the volume's years and the scan's look (the style handles), every
line cited as ``<Sv_Text_i>`` with special lines placed in words, then FLUX.3's layout JSON
(``[y_min, x_min, y_max, x_max]`` on 0-1000, y first) quoting each line's transcription.

Input: the Transkribus-export layout of the GT folders, ``<root>/<collection>/<volume>/<volume>/`` with the
scans and a ``page/`` folder of PAGE XML named like them. Output: an imagefolder dataset per split
(``<out>/<split>/<id>.jpg`` + ``metadata.jsonl``, one ``CropRecord`` per crop), loadable with
``datasets.load_dataset("<out>/<split>")``.

    uv run --no-sync python -m page_synth.data.build_crops \
        --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --max-pages 40 --out-dir data_flux3/layout_crops_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

from PIL import Image
from pydantic import BaseModel

from page_synth.caption import ROLE_NAMES, caption_for
from page_synth.data.page_xml import iter_local, local_name, page_line_text, parse_points
from page_synth.geometry import CropSettings, Line, Region, crops_for_page
from page_synth.look import scan_look
from page_synth.provenance import Provenance, provenance_of
from page_synth.records import CropRecord

Image.MAX_IMAGE_PIXELS = None  # archival spreads exceed PIL's decompression-bomb default


SPLITS = ("train", "val")
SCAN_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class Page(BaseModel):
    """A ground-truth page with a scan and transcribed text."""

    collection: str
    path: Path  # the PAGE XML, relative to the GT root
    scan: Path
    regions: list[Region]

    @property
    def id(self) -> str:
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]

    @property
    def provenance(self) -> Provenance:
        return provenance_of(self.collection, self.path)


def parse_page(xml_path: Path) -> list[Region]:
    """A PAGE XML page's text regions in reading order, each with its transcribed lines from top to bottom."""
    page = iter_local(ET.parse(xml_path).getroot(), "Page")[0]
    ordered: list[tuple[int, Region]] = []
    for position, element in enumerate(iter_local(page, "TextRegion")):
        role = _region_role(element)
        parsed = (_parse_line(child, role) for child in element if local_name(child.tag) == "TextLine")
        lines = sorted((line for line in parsed if line is not None), key=lambda line: line.baseline_y)
        if lines:
            ordered.append((_reading_order(element, position), Region(lines=lines)))
    return [region for _, region in sorted(ordered, key=lambda item: item[0])]


def _region_role(region: ET.Element) -> str:
    structure = re.search(r"structure \{type:([^;}]+)", region.attrib.get("custom", ""))
    role = region.attrib.get("type") or (structure.group(1) if structure else "")
    return role if role in ROLE_NAMES else ""


def _reading_order(region: ET.Element, position: int) -> int:
    """The region's Transkribus reading-order index, else its position in the file."""
    match = re.search(r"readingOrder \{index:(\d+);", region.attrib.get("custom", ""))
    return int(match.group(1)) if match else position


def _parse_line(element: ET.Element, role: str) -> Line | None:
    """A TextLine with its transcription, outline and baseline height; None without text or outline."""
    text = (page_line_text(element) or "").strip()
    coords = _child(element, "Coords")
    if not text or coords is None:
        return None
    polygon = parse_points(coords.attrib.get("points", ""))
    if len(polygon) < 3:
        return None
    baseline = _child(element, "Baseline")
    points = parse_points(baseline.attrib.get("points", "")) if baseline is not None else []
    baseline_y = statistics.mean(y for _, y in points) if points else max(y for _, y in polygon)
    return Line(polygon=polygon, baseline_y=baseline_y, text=text, role=role)


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if local_name(child.tag) == name), None)


def sample_pages(collection_dir: Path, count: int, rng: random.Random) -> list[Page]:
    """Up to ``count`` random pages of a collection with a scan and transcribed text."""
    xml_paths = sorted(p for p in collection_dir.rglob("page/*.xml") if not p.name.startswith("._"))  # "._*": macOS
    rng.shuffle(xml_paths)
    pages: list[Page] = []
    for xml_path in xml_paths:
        if len(pages) == count:
            break
        page = _load_page(collection_dir, xml_path)
        if page is not None:
            pages.append(page)
    return pages


def _load_page(collection_dir: Path, xml_path: Path) -> Page | None:
    scan = _find_scan(xml_path)
    if scan is None:
        return None
    regions = parse_page(xml_path)
    if not regions:
        return None
    return Page(
        collection=collection_dir.name,
        path=xml_path.relative_to(collection_dir.parent),
        scan=scan,
        regions=regions,
    )


def _find_scan(xml_path: Path) -> Path | None:
    """The scan a PAGE XML in ``<volume>/page/`` describes: the same stem, one folder up."""
    candidates = (xml_path.parent.parent / f"{xml_path.stem}{suffix}" for suffix in SCAN_SUFFIXES)
    return next((path for path in candidates if path.is_file()), None)


def split_of(page: Page, val_frac: float) -> str:
    """The page's split, by its hash, so a page's crops never straddle train and val."""
    return "val" if int(page.id, 16) % 1000 < val_frac * 1000 else "train"


def write_crops(page: Page, split_dir: Path, settings: CropSettings) -> list[CropRecord]:
    """Cut a page into crops, save them as JPEGs in ``split_dir`` and return their metadata records."""
    with Image.open(page.scan) as img:
        scan = img.convert("RGB")
    records: list[CropRecord] = []
    for k, crop in enumerate(crops_for_page(page.regions, scan.size, settings)):
        left, top, right, bottom = (round(v) for v in crop.rect)
        tile = scan.crop((left, top, right, bottom)).resize(crop.size, Image.Resampling.LANCZOS)
        name = f"{page.collection}_{page.id}_{k:02d}.jpg"
        tile.save(split_dir / name, quality=95)
        look = scan_look(tile)
        caption = caption_for(crop, page.provenance, look)
        records.append(
            CropRecord(
                file_name=name,
                caption=caption.text,
                collection=page.collection,
                page=str(page.path),
                width=crop.size[0],
                height=crop.size[1],
                **look.model_dump(),
                layout=caption.layout,
                lines=caption.lines,
            )
        )
    return records


def _create_metadata(split_dir: Path) -> TextIO:
    split_dir.mkdir(parents=True, exist_ok=True)
    return open(split_dir / "metadata.jsonl", "w", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True, help="GT folder with one subfolder per collection")
    ap.add_argument("--collections", nargs="*", default=None, help="collection names (default: all)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=40, help="pages sampled per collection")
    ap.add_argument("--val-frac", type=float, default=0.05, help="pages held out (by page hash)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--line-spacing", type=float, help="target line pitch in px")
    ap.add_argument("--min-line-spacing", type=float, help="skip crops scaled below this pitch")
    ap.add_argument("--min-merge-spacing", type=float, help="smallest pitch a margin note may shrink its column to")
    ap.add_argument("--min-width", type=int, help="narrower regions are only cropped together with a wide one")
    ap.add_argument("--min-lines", type=int)
    ap.add_argument("--overlap-lines", type=int, help="lines shared by consecutive windows of a column")
    ap.set_defaults(**CropSettings().model_dump())
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    settings = CropSettings.model_validate(vars(args))
    root = Path(args.gt_root).expanduser()
    out = Path(args.out_dir)
    rng = random.Random(args.seed)
    totals: Counter[str] = Counter()
    with ExitStack() as stack:
        metadata = {split: stack.enter_context(_create_metadata(out / split)) for split in SPLITS}
        for collection in args.collections or sorted(p.name for p in root.iterdir() if p.is_dir()):
            for page in sample_pages(root / collection, args.max_pages, rng):
                split = split_of(page, args.val_frac)
                records = write_crops(page, out / split, settings)
                metadata[split].writelines(f"{record.model_dump_json()}\n" for record in records)
                totals.update(pages=1, crops=len(records), lines=sum(len(record.lines) for record in records))
            print(f"{collection}: {dict(totals)}", flush=True)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
