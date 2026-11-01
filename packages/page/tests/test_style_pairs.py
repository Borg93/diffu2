"""The style pairs: the manifest names each pair's target and reference, a same-hand reference shares no line with its
target, and a build never writes into a dir that holds an earlier one. (A pair's caption is the one a crop drawn from
its reference gets: test_plan.)"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from diffu_page.captions.hand import HandLook, ReferenceLook
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.data.style_pairs import Pair, Scan, app, by_volume, pick_pairs, pick_same_hand, training_pairs
from diffu_page.page.crops.geometry import Crop

if TYPE_CHECKING:
    from conftest import RecordMaker

LOOK = ReferenceLook(  # every reference's look as measured
    scan=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
    hand=HandLook(slant=33.0, stroke=0.05, letters=0.3, density=0.2, paper="#D8C8A8", ink="#4A3828"),
)


@pytest.fixture
def make_scan(body_and_notes_crop: Crop, make_record: RecordMaker) -> Callable[[int, list[str]], Scan]:
    """Makes a crop of scan ``page`` of one volume holding lines with ``texts``: ``make_scan(page, texts)``."""

    def make(page: int, texts: list[str]) -> Scan:
        lines = [
            LineRecord(id=f"Sv_Text_{k}", text=text, role="", bbox=[0, 0, 100, 1000]) for k, text in enumerate(texts)
        ]
        record = make_record(body_and_notes_crop, f"{page}_{texts[0]}.jpg", page=f"40004026_{page:05d}.xml")
        return Scan(record=record.model_copy(update={"lines": lines}), path=Path(record.file_name))

    return make


def _pairs(make_scan: Callable[[int, list[str]], Scan]) -> list[Pair]:
    """Six crops on three facing pages of one volume, each crop's lines its own, paired with same-hand references."""
    scans = [make_scan(10 + k // 2, [f"line {k}"]) for k in range(6)]
    return pick_pairs(scans, by_volume(scans), rng=random.Random(0))


def test_training_pairs_name_each_pairs_target_and_reference(make_scan: Callable[[int, list[str]], Scan]) -> None:
    pairs = _pairs(make_scan)

    built = training_pairs(pairs, {pair.reference.path: LOOK for pair in pairs})

    assert len(pairs) == 6
    assert [(row.media[0].file_name, row.media[1].file_name, row.metadata.reference) for row in built.rows] == [
        (
            f"targets/{pair.target.record.file_name}",
            f"refs/{pair.reference.record.file_name}",
            pair.reference.record.file_name,
        )
        for pair in pairs
    ]


def test_same_hand_reference_is_a_crop_of_the_page_sharing_no_line(make_scan: Callable[[int, list[str]], Scan]) -> None:
    target, overlapping, apart, facing = (
        make_scan(10, ["a", "b"]),
        make_scan(10, ["b", "c"]),
        make_scan(10, ["d"]),
        make_scan(11, ["e"]),
    )

    reference = pick_same_hand(target, by_volume([target, overlapping, apart, facing]), random.Random(0))

    assert reference is apart


def test_same_hand_reference_falls_back_to_a_facing_page_crop_sharing_no_line(
    make_scan: Callable[[int, list[str]], Scan],
) -> None:
    """Consecutive scans can overlap (one sheet photographed in two parts), so a crop of the facing scan may hold the
    target's own lines."""
    target, overlapped, facing, farther = (
        make_scan(20, ["a", "b"]),
        make_scan(21, ["b", "c"]),
        make_scan(21, ["d"]),
        make_scan(23, ["e"]),
    )
    volumes = by_volume([target, overlapped, facing, farther])

    references = [pick_same_hand(target, volumes, random.Random(seed)) for seed in range(8)]

    assert references == [facing] * 8


def test_a_training_sets_spread_keeps_only_the_files_its_pairs_name(
    make_scan: Callable[[int, list[str]], Scan],
) -> None:
    pairs = _pairs(make_scan)
    built = training_pairs(pairs, {pair.reference.path: LOOK for pair in pairs})

    spread = built.spread(2)

    assert [row.metadata.row_index for row in spread.rows] == [0, 1]
    assert {p.destination.as_posix() for p in spread.placements} == {
        media.file_name for row in spread.rows for media in row.media
    }


def test_style_pairs_refuses_to_build_into_a_dir_that_holds_an_earlier_build(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "train.jsonl").write_text("")

    result = CliRunner().invoke(app, ["--crops", str(tmp_path / "crops"), "--out", str(tmp_path / "out")])

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2
    assert "is not empty: an earlier build's files would stay beside this one's" in message
