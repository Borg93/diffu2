"""A pool's scans written to the page table from their ALTO files, and the table resumed."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lance
import pytest

from diffu_curate.page_features import longest_word_run, repeated_lines, write_pool_features
from diffu_curate.settings import Pool

LINES = [
    ("den 3 April", (100, 100, 400, 50)),
    ("Dito", (100, 160, 400, 50)),
    ("Dito", (100, 220, 400, 50)),
    ("samma tid till användande", (100, 280, 500, 50)),
    ("samma tid till användande", (100, 340, 500, 50)),
]


def rows(table: Path) -> dict[str, dict[str, object]]:
    return {row["scan"]: row for row in lance.dataset(str(table)).to_table().to_pylist()}


def test_write_pool_features_writes_a_row_for_every_scan(
    pool: Pool, add_scan: Callable[[str, str], Path], alto: Callable[..., str], tmp_path: Path
) -> None:
    add_scan("V1_0001", alto("htrflow", LINES, confidence=0.9911))
    add_scan("V1_0002", "<alto><broken")
    add_scan("V2_0001", alto("Transkribus", []))
    table = tmp_path / "pages.lance"

    list(write_pool_features(pool, table, workers=1))

    written = rows(table)
    assert set(written) == {"V1_0001", "V1_0002", "V2_0001"}
    page = written["V1_0001"]
    assert {k: page[k] for k in ("pool", "volume", "source", "regions", "lines", "empty_lines", "chars", "words")} == {
        "pool": "pool", "volume": "V1", "source": "htrflow", "regions": 1, "lines": 5, "empty_lines": 0,
        "chars": 73, "words": 13,
    }  # fmt: skip
    assert page["page_confidence"] == pytest.approx(0.9911)
    assert page["repeated_lines"] == 2  # both pairs: Dito is 4 characters, the shortest length counted
    assert page["odd_density_lines"] == 2  # the two Dito: 4 characters on lines that hold 11 to 25
    assert page["polygon_cover"] == pytest.approx((3 * 400 * 50 + 2 * 500 * 50) / (1000 * 800))
    assert written["V1_0002"]["source"] == "unreadable"
    assert "V1_0002.xml" in str(written["V1_0002"]["error"])
    assert (written["V2_0001"]["source"], written["V2_0001"]["lines"]) == ("Transkribus", 0)


def test_write_pool_features_leaves_a_volume_the_table_holds(
    pool: Pool, add_scan: Callable[[str, str], Path], alto: Callable[..., str], tmp_path: Path
) -> None:
    add_scan("V1_0001", alto("htrflow", LINES))
    table = tmp_path / "pages.lance"
    list(write_pool_features(pool, table, workers=1))
    add_scan("V1_0002", alto("htrflow", LINES))
    add_scan("V2_0001", alto("htrflow", LINES))

    list(write_pool_features(pool, table, workers=1))

    assert sorted(rows(table)) == ["V1_0001", "V2_0001"]
    assert lance.dataset(str(table)).count_rows() == 2


@pytest.mark.parametrize(
    ("lines", "count"),
    [
        pytest.param(["Dito", "Dito", "Dito"], 2, id="at-the-shortest-length-counted"),
        pytest.param(["24.", "24.", "24."], 0, id="shorter-than-counted"),
        pytest.param(["en rad", "en annan", "en rad"], 0, id="not-next-to-each-other"),
    ],
)
def test_repeated_lines_counts_lines_saying_the_line_before(lines: list[str], count: int) -> None:
    assert repeated_lines(lines) == count


@pytest.mark.parametrize(
    ("words", "run"),
    [
        pytest.param([], 0, id="no-words"),
        pytest.param(["och"], 1, id="one-word"),
        pytest.param(["do", "do", "do", "och", "do"], 3, id="a-run-then-another"),
    ],
)
def test_longest_word_run(words: list[str], run: int) -> None:
    assert longest_word_run(words) == run
