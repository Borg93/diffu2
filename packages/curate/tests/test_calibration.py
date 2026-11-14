"""A pool's old labels measured against a ground truth on disk."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lance
import pytest

from diffu_curate.calibration import calibrate, write_calibration
from diffu_curate.settings import Pool

TRUE = [
    ("Anno 1717 den 3 Maij", (100, 100, 600, 60)),
    ("Närwarande Herrar", (100, 180, 500, 60)),
    ("en rad som missades", (100, 600, 500, 60)),
]
OLD = [
    ("Anno 1711 den 3 Maij", (100, 100, 600, 60)),  # one digit wrong
    ("Närwarande Herrar", (110, 185, 480, 50)),  # exact, its box a little smaller
    ("på tomt papper", (700, 700, 200, 40)),  # where the truth has no line
]


@pytest.fixture
def ground_truth(tmp_path: Path, alto: Callable[..., str]) -> Path:
    """A ground truth of one collection: V1's scans 0001 (a training scan) and 0002 (named by its test list)."""
    folder = tmp_path / "gt" / "GT-RA_test" / "V1 volym" / "V1 volym" / "alto"
    folder.mkdir(parents=True)
    for stem in ("V1_0001", "V1_0002", "V1_0003"):
        (folder / f"{stem}.xml").write_text(alto("Transkribus", TRUE), encoding="utf-8")
    (folder / "test_0050").write_text("V1_0002.xml\n", encoding="utf-8")
    return tmp_path / "gt"


def test_calibrate_measures_the_old_label_against_the_truth(
    pool: Pool, add_scan: Callable[[str, str], Path], alto: Callable[..., str], ground_truth: Path, tmp_path: Path
) -> None:
    add_scan("V1_0001", alto("htrflow", OLD))
    add_scan("V1_0002", alto("htrflow", OLD))
    add_scan("V1_0003", alto("Transkribus", TRUE))  # the pool holds a transcriber's export of this scan: truth already

    result = calibrate(ground_truth, pool, "test_0050")
    write_calibration(result, tmp_path / "calibration")

    assert [(s.scan, s.split, s.source) for s in result.scans] == [
        ("V1_0001", "train", "htrflow"),
        ("V1_0002", "test", "htrflow"),
        ("V1_0003", "train", "Transkribus"),
    ]
    page = result.pages[0]
    assert (page.scan, page.old_lines, page.true_lines, page.matched, page.missed, page.astray) == (
        "V1_0001", 3, 3, 2, 1, 1,
    )  # fmt: skip
    assert page.cer_matched == pytest.approx(1 / (20 + 17))
    wrong = next(line for line in result.lines if line.scan == "V1_0001" and line.errors)
    assert (wrong.old, wrong.errors, wrong.numbers_differ, wrong.iou) == ("Anno 1711 den 3 Maij", 1, True, 1.0)
    assert lance.dataset(str(tmp_path / "calibration" / "pages.lance")).count_rows() == 2
    assert lance.dataset(str(tmp_path / "calibration" / "lines.lance")).count_rows() == 4
