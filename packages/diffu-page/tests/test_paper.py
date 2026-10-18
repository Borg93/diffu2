"""Tests for the paper-background corpus builder — the content mask finds ink (not clean paper), and
``extract_paper_crops`` yields blank, inpainted crops from a scan with a stroke on it."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from diffu_page.paper import content_mask, extract_paper_crops


def _synthetic_scan(h: int = 240, w: int = 320) -> np.ndarray:
    """Aged-paper gray with one dark handwriting-like stroke band across the middle."""
    img = np.full((h, w, 3), 210, np.uint8)
    img[100:106, 40:280] = 30
    return img


def test_content_mask_marks_the_stroke_and_leaves_paper_clean():
    gray = cv2.cvtColor(_synthetic_scan(), cv2.COLOR_BGR2GRAY)
    mask = content_mask(gray)
    assert mask.shape == gray.shape
    assert mask[100:106, 60:260].mean() > 0.9, "the dark stroke must be masked as content"
    assert mask[:40, :40].sum() == 0, "clean paper corners must not be masked"


def test_extract_paper_crops_writes_a_blank_inpainted_crop(tmp_path: Path):
    scan_dir = tmp_path / "scans"
    out_dir = tmp_path / "crops"
    scan_dir.mkdir()
    cv2.imwrite(str(scan_dir / "scan_a.png"), _synthetic_scan())

    n = extract_paper_crops(str(scan_dir), str(out_dir), n_per_scan=1)

    assert n == 1
    files = list(out_dir.glob("*.png"))
    assert len(files) == 1 and files[0].name.startswith("scan_a")
    crop = cv2.imread(str(files[0]), cv2.IMREAD_GRAYSCALE)
    assert crop is not None and crop.size > 0
    dark_frac = float((crop < 120).mean())
    assert dark_frac < 0.005, f"residual stroke must be inpainted out (dark frac {dark_frac:.4f})"
    assert crop.mean() >= 190, "the crop must read as sane paper tone, not a dark scan"


def test_extract_paper_crops_rejects_ink_heavy_windows(tmp_path: Path):
    scan_dir = tmp_path / "scans"
    out_dir = tmp_path / "crops"
    scan_dir.mkdir()
    inky = _synthetic_scan()
    inky[20:220, 20:300] = 25  # a page-filling text block: every window exceeds max_ink_frac
    cv2.imwrite(str(scan_dir / "full.png"), inky)

    assert extract_paper_crops(str(scan_dir), str(out_dir), n_per_scan=2) == 0
    assert not list(out_dir.glob("*.png"))
