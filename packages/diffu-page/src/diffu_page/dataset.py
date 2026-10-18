"""Dataset assembly — line-crop export, a per-page manifest, and deterministic train/val/test splits.

A full-page PNG + label file is one output shape; many HTR trainers (TrOCR, PyLaia) instead consume
CROPPED line images paired with their transcription. This module turns composed pages into that line-crop
dataset and writes a manifest, so a generated corpus is reproducible, auditable, and splittable. A page and
all of its line crops land wholly in one split — never straddling train/val/test — so there is no leakage.

Splitting is a pure function of the page index (a fixed hash), so the split assignment is stable across
runs and independent of how many pages you generate.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from .labels import LineLabel, Point

_KNUTH = 2654435761  # Knuth multiplicative hash constant -> a stable, well-spread split from the index


class CropRecord(BaseModel):
    """One exported line crop: where its image is + what it says + which page/split it came from."""

    image: str  # path relative to the output dir
    text: str
    page: str
    region: str
    order: int
    split: str


class PageRecord(BaseModel):
    """One page's manifest row — enough to reproduce and audit it without opening the image."""

    image: str
    width: int
    height: int
    template: str
    seed: int
    binding: bool
    recto: bool
    n_lines: int
    split: str
    styles: list[str] = Field(default_factory=list)


def assign_split(index: int, *, val_frac: float, test_frac: float) -> str:
    """Deterministic train/val/test split for page ``index`` — a fixed hash so it's stable across runs and
    independent of the page count. ``val_frac``/``test_frac`` are the target fractions; the rest is train."""
    r = ((index * _KNUTH) & 0xFFFFFFFF) / 0xFFFFFFFF
    if r < test_frac:
        return "test"
    if r < test_frac + val_frac:
        return "val"
    return "train"


def _bbox(polygon: list[Point]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def line_crops(
    page: np.ndarray, lines: list[LineLabel], *, pad: int = 6
) -> list[tuple[np.ndarray, LineLabel]]:
    """Crop each line's (padded) polygon bbox from the page, in reading order. Degenerate/off-page boxes
    are skipped. Returns ``(crop_bgr, line)`` pairs — the caller owns naming + the transcription side file."""
    h, w = page.shape[:2]
    out: list[tuple[np.ndarray, LineLabel]] = []
    for line in sorted(lines, key=lambda ln: ln.order):
        if not line.polygon:
            continue
        x0, y0, x1, y1 = _bbox(line.polygon)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append((page[y0:y1, x0:x1].copy(), line))
    return out


def export_line_crops(
    page: np.ndarray, lines: list[LineLabel], *, out_dir: Path, page_name: str, split: str, pad: int = 6
) -> list[CropRecord]:
    """Write every line crop of one page under ``out_dir/lines/`` and return its manifest records."""
    crops_dir = out_dir / "lines"
    crops_dir.mkdir(parents=True, exist_ok=True)
    records: list[CropRecord] = []
    for crop, line in line_crops(page, lines, pad=pad):
        rel = f"lines/{page_name}_l{line.order:03d}.png"
        cv2.imwrite(str(out_dir / rel), crop)
        records.append(
            CropRecord(
                image=rel, text=line.text, page=page_name, region=line.region, order=line.order, split=split
            )
        )
    return records


def write_manifest(out_dir: Path, pages: list[PageRecord], crops: list[CropRecord]) -> None:
    """Write the corpus manifest: one JSONL row per page (``manifest.jsonl``), and — when line crops were
    exported — one JSONL row per crop (``lines.jsonl``)."""
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(p.model_dump_json() for p in pages) + "\n", encoding="utf-8"
    )
    if crops:
        (out_dir / "lines.jsonl").write_text(
            "\n".join(c.model_dump_json() for c in crops) + "\n", encoding="utf-8"
        )
