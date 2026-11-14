"""Every page of the Svea pretraining pool as a row of numbers read from its ALTO file alone: who wrote the file, the
page's size and confidence, how many regions, lines and characters it holds, how tall and how dense its lines are, how
many lines hold far less or far more text than the page's lines usually do for their width, how much of the page its
line polygons cover, and how repetitive its text is. --pool is the folder of the pool's volumes of ALTO files (the
Svea pool by default; /mnt/work/haas/output/andraarkiv for the other archives). Rows are appended to one Lance table (--out), each
with its pool's name (--name); a volume the table already holds is skipped, so a stopped run goes on where it was, and
the scores of later tests (page embeddings, Jev, backbone overlap) are added to the same table as columns.

    uv run python packages/htr_train/lion_data_prep/curate/features.py --name svea --out runs/htr/curate/pages.lance --workers 24
    uv run python packages/htr_train/lion_data_prep/curate/features.py --name andraarkiv --pool /mnt/work/haas/output/andraarkiv --out runs/htr/curate/pages.lance --workers 24
"""

from __future__ import annotations

import argparse
import statistics
import xml.etree.ElementTree as ET
import zlib
from multiprocessing import Pool
from pathlib import Path

import lance
import pyarrow as pa

NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
POOL = Path("/mnt/work/haas/deliveries_old/svea-all")
VOLUMES_A_WRITE = 200  # volumes gathered before the table is appended to, so a stopped run loses at most these
ODD_DENSITY = (0.5, 2.0)  # a line is odd when its characters per pixel are outside these multiples of its page's median


def number(element: ET.Element, name: str) -> float:
    try:
        return float(element.get(name, 0))
    except ValueError:
        return 0.0


def polygon_area(text: str) -> float:
    numbers = [float(n) for n in text.replace(",", " ").split()]
    xs, ys = numbers[0::2], numbers[1::2]
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    return abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(len(xs)))) / 2


def repeated_lines(lines: list[str]) -> int:
    """How many lines say what the line before them says."""
    return sum(1 for a, b in zip(lines, lines[1:], strict=False) if a == b and len(a) > 3)


def longest_word_run(words: list[str]) -> int:
    """The longest run of one word said again and again."""
    longest = run = 1 if words else 0
    for a, b in zip(words, words[1:], strict=False):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    return longest


def page_row(path: Path) -> dict[str, object]:
    row: dict[str, object] = {"volume": path.parent.name, "scan": path.stem}
    try:
        text = path.read_text(encoding="utf-8")
        root = ET.fromstring(text)
    except (OSError, ET.ParseError, UnicodeDecodeError) as error:
        return {**row, "source": "unreadable", "error": str(error)[:200]}
    head = text[:3000]
    row["source"] = "htrflow" if "htrflow" in head else "transkribus" if "Transkribus" in head else "other"
    page = root.find(".//a:Page", NS)
    if page is None:
        return {**row, "error": "no Page"}
    width, height = number(page, "WIDTH"), number(page, "HEIGHT")
    confidence = page.get("PC")
    texts, heights, widths, densities = [], [], [], []
    area = 0.0
    empty = 0
    for line in page.iterfind(".//a:TextLine", NS):
        content = " ".join(s.get("CONTENT", "") for s in line.iterfind("a:String", NS)).strip()
        if not content:
            empty += 1
            continue
        line_width, line_height = number(line, "WIDTH"), number(line, "HEIGHT")
        polygon = line.find("a:Shape/a:Polygon", NS)
        area += polygon_area(polygon.get("POINTS", "")) if polygon is not None else line_width * line_height
        texts.append(content)
        heights.append(line_height)
        widths.append(line_width)
        densities.append(len(content) / max(1.0, line_width))
    joined = "\n".join(texts)
    median_density = statistics.median(densities) if densities else 0.0
    odd = [d for d in densities if median_density and not ODD_DENSITY[0] <= d / median_density <= ODD_DENSITY[1]]
    quartiles = statistics.quantiles(heights, n=4) if len(heights) >= 4 else None
    encoded = joined.encode("utf-8")
    return {
        **row,
        "width": width,
        "height": height,
        "page_confidence": float(confidence) if confidence else None,
        "regions": sum(1 for _ in page.iterfind(".//a:TextBlock", NS)),
        "lines": len(texts),
        "empty_lines": empty,
        "chars": len(joined),
        "words": len(joined.split()),
        "digit_share": sum(c.isdigit() for c in joined) / max(1, len(joined)),
        "line_height_median": statistics.median(heights) if heights else 0.0,
        "line_height_spread": (quartiles[2] - quartiles[0]) / max(1.0, quartiles[1]) if quartiles else None,
        "line_width_median": statistics.median(widths) if widths else 0.0,
        "chars_per_100px_median": 100 * median_density,
        "odd_density_lines": len(odd),
        "polygon_cover": area / max(1.0, width * height),
        "repeated_lines": repeated_lines(texts),
        "longest_word_run": longest_word_run(joined.split()),
        "compression": len(zlib.compress(encoded)) / len(encoded) if len(encoded) >= 200 else None,
    }


SCHEMA = pa.schema(
    [
        pa.field("pool", pa.utf8(), nullable=False),
        pa.field("volume", pa.utf8(), nullable=False),
        pa.field("scan", pa.utf8(), nullable=False),
        pa.field("source", pa.utf8()),
        pa.field("error", pa.utf8()),
        pa.field("width", pa.float32()),
        pa.field("height", pa.float32()),
        pa.field("page_confidence", pa.float32()),
        pa.field("regions", pa.int32()),
        pa.field("lines", pa.int32()),
        pa.field("empty_lines", pa.int32()),
        pa.field("chars", pa.int32()),
        pa.field("words", pa.int32()),
        pa.field("digit_share", pa.float32()),
        pa.field("line_height_median", pa.float32()),
        pa.field("line_height_spread", pa.float32()),
        pa.field("line_width_median", pa.float32()),
        pa.field("chars_per_100px_median", pa.float32()),
        pa.field("odd_density_lines", pa.int32()),
        pa.field("polygon_cover", pa.float32()),
        pa.field("repeated_lines", pa.int32()),
        pa.field("longest_word_run", pa.int32()),
        pa.field("compression", pa.float32()),
    ]
)


def volume_rows(volume: Path) -> list[dict[str, object]]:
    return [page_row(path) for path in sorted(volume.glob("*.xml"))]


def written_volumes(table: Path, pool: str) -> set[str]:
    """The pool's volumes the table already holds."""
    if not table.exists():
        return set()
    held = lance.dataset(str(table)).to_table(columns=["volume"], filter=f"pool = '{pool}'")
    return set(held.column("volume").to_pylist())


def append(table: Path, pool: str, rows: list[dict[str, object]]) -> None:
    columns = {f.name: [pool if f.name == "pool" else row.get(f.name) for row in rows] for f in SCHEMA}
    lance.write_dataset(pa.table(columns, schema=SCHEMA), str(table), mode="append" if table.exists() else "create")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--name", required=True, help="the pool's name in the table: svea, andraarkiv")
    parser.add_argument("--out", type=Path, required=True, help="the Lance table")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    held = written_volumes(args.out, args.name)
    volumes = [v for v in sorted(args.pool.iterdir()) if v.is_dir() and v.name not in held]
    pending: list[dict[str, object]] = []
    done = 0
    with Pool(args.workers) as pool:
        for count, rows in enumerate(pool.imap_unordered(volume_rows, volumes), 1):
            pending += rows
            if count % VOLUMES_A_WRITE == 0 or count == len(volumes):
                append(args.out, args.name, pending)
                done += len(pending)
                pending = []
                print(f"{count} of {len(volumes)} volumes, {done} pages", flush=True)
    if args.out.exists():
        lance.dataset(str(args.out)).optimize.compact_files()
    print(f"done: {done} pages written this run ({len(held)} volumes were already in the table)", flush=True)


if __name__ == "__main__":
    main()
