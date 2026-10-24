"""The v3 reference pairs: the manifest and the eval rows name each pair's target and reference, a same-hand
reference shares no line with its target, and a build never writes into a dir that holds an earlier one."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.records import CropRecord
from diffu_page.captions.restyle import CaptionParts, edit_caption
from diffu_page.crops.geometry import Crop
from diffu_page.data.style_pairs import (
    ExamplePair,
    Pair,
    Scan,
    app,
    by_example_picks,
    by_volume,
    eval_set,
    pick_pairs,
    pick_reference,
    pick_same_hand,
    training_pairs,
)


def _scans(make_record: Callable[..., CropRecord], crop: Crop, directory: Path) -> list[Scan]:
    """Six crops on three pages of one volume, as build_crops writes them."""
    directory.mkdir()
    scans: list[Scan] = []
    for k in range(6):
        record = make_record(crop, f"{k}.jpg", page=f"{k // 2:04d}.xml")
        Image.new("RGB", crop.size, "white").save(directory / record.file_name)
        scans.append(Scan(record=record, path=directory / record.file_name))
    return scans


def _edit_caption(pair: Pair) -> str:
    return edit_caption(CaptionParts.of(pair.target.record))


def test_training_pairs_name_each_pairs_target_and_reference(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord], tmp_path: Path
) -> None:
    scans = _scans(make_record, body_and_notes_crop, tmp_path / "crops")
    pairs = pick_pairs(scans, by_volume(scans), rng=random.Random(0), pick=pick_reference)

    built = training_pairs(pairs, _edit_caption)

    assert [(row.media[0].file_name, row.media[1].file_name, row.metadata.reference) for row in built.rows] == [
        (
            f"targets/{pair.target.record.file_name}",
            f"refs/{pair.reference.record.file_name}",
            pair.reference.record.file_name,
        )
        for pair in pairs
    ]


@pytest.fixture
def make_scan(body_and_notes_crop: Crop, make_record: Callable[..., CropRecord]) -> Callable[[int, list[str]], Scan]:
    """Makes a crop of scan ``page`` of one volume holding lines with ``texts``: ``make_scan(page, texts)``."""

    def make(page: int, texts: list[str]) -> Scan:
        lines = [
            LineRecord(id=f"Sv_Text_{k}", text=text, role="", bbox=[0, 0, 100, 1000]) for k, text in enumerate(texts)
        ]
        record = make_record(body_and_notes_crop, f"{page}_{texts[0]}.jpg", page=f"40004026_{page:05d}.xml")
        return Scan(record=record.model_copy(update={"lines": lines}), path=Path(record.file_name))

    return make


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
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord], tmp_path: Path
) -> None:
    scans = _scans(make_record, body_and_notes_crop, tmp_path / "crops")
    built = training_pairs(
        pick_pairs(scans, by_volume(scans), rng=random.Random(0), pick=pick_reference), _edit_caption
    )

    spread = built.spread(2)

    assert [row.metadata.row_index for row in spread.rows] == [0, 1]
    assert {p.destination.as_posix() for p in spread.placements} == {
        media.file_name for row in spread.rows for media in row.media
    }


def test_by_example_rows_name_their_reference_and_template(
    body_and_notes_crop: Crop, make_record: Callable[..., CropRecord], tmp_path: Path
) -> None:
    scans = _scans(make_record, body_and_notes_crop, tmp_path / "crops")
    picked = [
        ExamplePair(pair=Pair(target=scans[3], reference=scans[0]), template=3, label="a"),
        ExamplePair(pair=Pair(target=scans[1], reference=scans[4]), template=1, label="b"),
    ]

    built = eval_set(by_example_picks(picked), _edit_caption)

    assert [(row.file_name, row.reference, row.template) for row in built.rows] == [
        ("3.jpg", "refs/0.jpg", 3),
        ("1.jpg", "refs/4.jpg", 1),
    ]


def test_style_pairs_refuses_to_build_into_a_dir_that_holds_an_earlier_build(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "train.jsonl").write_text("")

    result = CliRunner().invoke(app, ["--crops", str(tmp_path / "crops"), "--out", str(tmp_path / "out")])

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2
    assert "is not empty: an earlier build's files would stay beside this one's" in message
