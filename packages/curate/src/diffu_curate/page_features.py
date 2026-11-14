"""What a scan's ALTO file alone says about its labels: one row a scan, for every scan of a pool.

Nothing here looks at the image, so these are cheap (minutes for millions of files) and blind to writing the old
pipeline did not find: a scan with no lines is as likely a ruled ledger its segmenter missed as an empty page.
"""

from __future__ import annotations

import statistics
import zlib
from collections.abc import Iterator
from multiprocessing import Pool as ProcessPool
from pathlib import Path

from pydantic import BaseModel

from diffu_curate.alto import AltoPage, read_alto
from diffu_curate.settings import Pool
from diffu_curate.table import append_rows, compact, held_values

ODD_DENSITY = (0.5, 2.0)  # a line is odd when its characters a pixel are outside these multiples of its page's median
REPEATED_LINE_CHARS = 4  # a shorter line said twice is a table's "Dito" or "24.", not the same line found twice
COMPRESSION_CHARS = 200  # shorter texts compress too little to compare
VOLUMES_A_WRITE = 200  # volumes gathered before the table is appended to: what a stopped run loses at most


class PageFeatures(BaseModel):
    """A scan's row: its pool, volume and name, who wrote its labels (``unreadable`` with the ``error`` when the file
    cannot be read), and what the labels hold. ``odd_density_lines`` counts lines with far less or far more text than
    the page's lines usually hold for their width, ``polygon_cover`` is the share of the scan its line outlines cover,
    ``repeated_lines`` counts lines saying what the line before says, and ``compression`` is the text's compressed size
    over its size (low: repetitive)."""

    pool: str
    volume: str
    scan: str
    source: str
    error: str | None = None
    width: float | None = None
    height: float | None = None
    page_confidence: float | None = None
    regions: int | None = None
    lines: int | None = None
    empty_lines: int | None = None
    chars: int | None = None
    words: int | None = None
    digit_share: float | None = None
    line_height_median: float | None = None
    line_height_spread: float | None = None
    line_width_median: float | None = None
    chars_per_100px_median: float | None = None
    odd_density_lines: int | None = None
    polygon_cover: float | None = None
    repeated_lines: int | None = None
    longest_word_run: int | None = None
    compression: float | None = None


def outline_area(outline: tuple[tuple[float, float], ...]) -> float:
    """The area a closed outline encloses (the shoelace formula)."""
    return abs(sum(x * outline[i - 1][1] - outline[i - 1][0] * y for i, (x, y) in enumerate(outline))) / 2


def repeated_lines(lines: list[str]) -> int:
    """How many lines say what the line before them says, short ones aside (:data:`REPEATED_LINE_CHARS`)."""
    return sum(1 for a, b in zip(lines, lines[1:], strict=False) if a == b and len(a) >= REPEATED_LINE_CHARS)


def longest_word_run(words: list[str]) -> int:
    """The longest run of one word said again and again; 0 for no words."""
    longest = run = min(len(words), 1)
    for a, b in zip(words, words[1:], strict=False):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    return longest


def odd_density_lines(densities: list[float]) -> int:
    """How many of a page's lines hold far less or far more text for their width than its median line."""
    median = statistics.median(densities) if densities else 0.0
    if not median:
        return 0
    return sum(1 for density in densities if not ODD_DENSITY[0] <= density / median <= ODD_DENSITY[1])


def features_of(page: AltoPage, pool: str, volume: str, scan: str) -> PageFeatures:
    """The row of a scan whose labels are ``page``."""
    texts = [line.text for line in page.lines]
    text = "\n".join(texts)
    encoded = text.encode("utf-8")
    heights = [line.height for line in page.lines]
    quartiles = statistics.quantiles(heights, n=4) if len(heights) >= 4 else None
    densities = [len(line.text) / max(1.0, line.width) for line in page.lines]
    return PageFeatures(
        pool=pool,
        volume=volume,
        scan=scan,
        source=page.writer or "other",
        width=page.width,
        height=page.height,
        page_confidence=page.confidence,
        regions=page.regions,
        lines=len(texts),
        empty_lines=page.empty_lines,
        chars=len(text),
        words=len(text.split()),
        digit_share=sum(c.isdigit() for c in text) / max(1, len(text)),
        line_height_median=statistics.median(heights) if heights else 0.0,
        line_height_spread=(quartiles[2] - quartiles[0]) / max(1.0, quartiles[1]) if quartiles else None,
        line_width_median=statistics.median(line.width for line in page.lines) if page.lines else 0.0,
        chars_per_100px_median=100 * statistics.median(densities) if densities else 0.0,
        odd_density_lines=odd_density_lines(densities),
        polygon_cover=sum(outline_area(line.outline) for line in page.lines) / max(1.0, page.width * page.height),
        repeated_lines=repeated_lines(texts),
        longest_word_run=longest_word_run(text.split()),
        compression=len(zlib.compress(encoded)) / len(encoded) if len(encoded) >= COMPRESSION_CHARS else None,
    )


def scan_features(pool: str, path: Path) -> PageFeatures:
    """The row of the scan whose label file is ``path``; a file that cannot be read is a row with its error."""
    try:
        page = read_alto(path)
    except ValueError as error:
        return PageFeatures(
            pool=pool, volume=path.parent.name, scan=path.stem, source="unreadable", error=str(error)[:200]
        )
    return features_of(page, pool, path.parent.name, path.stem)


def _volume_features(job: tuple[str, Path]) -> list[PageFeatures]:
    pool, volume = job
    return [scan_features(pool, path) for path in sorted(volume.glob("*.xml"))]


def write_pool_features(pool: Pool, table: Path, workers: int) -> Iterator[tuple[int, int, int]]:
    """Append the row of every scan of ``pool`` to ``table``, a volume the table already holds left as it is, so a
    stopped run goes on where it was. Yields (volumes done, volumes to do, rows written) after each write."""
    held = held_values(table, "volume", f"pool = '{pool.name}'")
    volumes = [volume for volume in pool.volumes() if volume.name not in held]
    pending: list[PageFeatures] = []
    written = 0
    with ProcessPool(workers) as processes:
        jobs = ((pool.name, volume) for volume in volumes)
        for count, rows in enumerate(processes.imap_unordered(_volume_features, jobs), 1):
            pending += rows
            if count % VOLUMES_A_WRITE == 0 or count == len(volumes):
                append_rows(table, pending)
                written += len(pending)
                pending = []
                yield count, len(volumes), written
    compact(table)
