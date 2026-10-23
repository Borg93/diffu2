"""The evaluate CLI end to end on CPU (diffusers' tiny FLUX.3 test model, the real HTR read-back): every crop drawn
and scored into results.json, its hand measured beside its real crop's and its reference's, a crop that fails
recorded while the rest still run."""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.hand import HandDistance, HandLook, hand_distance, hand_look
from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.evaluate import EvalCrop, Hands, Results, app, carry_over, read_crops

LINES = [
    LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 500, 1000]),
    LineRecord(id="Sv_Text_2", text="den 4 maj", role="", bbox=[500, 0, 1000, 1000]),
]


def _eval_set(directory: Path, crops: list[EvalCrop]) -> Path:
    directory.mkdir()
    for crop in crops:
        Image.new("RGB", (crop.width, crop.height), "white").save(directory / crop.file_name)
    (directory / "metadata.jsonl").write_text("".join(f"{crop.model_dump_json()}\n" for crop in crops))
    return directory


REFERENCE_LINES = [[100, 50, 300, 950], [500, 50, 700, 950]]


def _crop(name: str, reference: str | None = None, reference_lines: list[list[int]] | None = None) -> EvalCrop:
    return EvalCrop(
        file_name=name,
        caption="A handwritten page.",
        width=64,
        height=32,
        lines=LINES,
        reference=reference,
        reference_lines=reference_lines,
    )


def _evaluate(tiny_flux3: Path, crops: Path, out: Path) -> int:
    args = ["--crops", str(crops), "--out", str(out), "--model", str(tiny_flux3), "-n", "2", "--steps", "2"]
    result = CliRunner().invoke(app, [*args, "--text-layer", "1", "--device", "cpu"])
    return result.exit_code


def test_evaluate_scores_every_crop_into_results(tiny_flux3: Path, tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg"), _crop("b.jpg")])

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "out")

    results = Results.model_validate_json((tmp_path / "out" / "results.json").read_text())
    assert exit_code == 0
    assert [(sample.index, sample.file) for sample in results.samples] == [(0, "a.jpg"), (1, "b.jpg")]
    assert results.failed == {}


def test_evaluate_records_a_failed_crop_and_scores_the_rest(tiny_flux3: Path, tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg"), _crop("b.jpg")])
    (crops / "b.jpg").unlink()  # its real crop can't be read back

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "out")

    results = Results.model_validate_json((tmp_path / "out" / "results.json").read_text())
    assert exit_code == 1
    assert [sample.file for sample in results.samples] == ["a.jpg"]
    assert list(results.failed) == [1]


def test_evaluate_measures_each_drawn_hand_beside_its_real_crop_and_its_reference(
    tiny_flux3: Path, tmp_path: Path
) -> None:
    crops = _eval_set(
        tmp_path / "crops",
        [_crop("a.jpg", "ref_a.png", REFERENCE_LINES), _crop("b.jpg", "ref_b.png", REFERENCE_LINES)],
    )
    references = [Image.linear_gradient("L").rotate(angle).convert("RGB") for angle in (30, 120)]
    for name, reference in zip(("ref_a.png", "ref_b.png"), references, strict=True):
        reference.save(crops / name)

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "out")

    results = Results.model_validate_json((tmp_path / "out" / "results.json").read_text())
    boxes = [line.bbox for line in LINES]
    drawn = [hand_look(Image.open(tmp_path / "out" / "generated" / f"{k:03d}.png"), boxes) for k in range(2)]
    real = hand_look(Image.new("RGB", (64, 32), "white"), boxes)
    reference_hands = [hand_look(reference, REFERENCE_LINES) for reference in references]
    chance = [hand_distance(drawn[0], reference_hands[1]), hand_distance(drawn[1], reference_hands[0])]
    assert exit_code == 0
    assert [sample.hands for sample in results.samples] == [
        Hands(
            drawn=drawn[k],
            real=real,
            reference=reference_hands[k],
            to_real=hand_distance(drawn[k], real),
            to_reference=hand_distance(drawn[k], reference_hands[k]),
            real_to_reference=hand_distance(real, reference_hands[k]),
        )
        for k in range(2)
    ]
    assert results.summary.hands is not None
    assert results.summary.hands.to_other_references == HandDistance.model_validate(
        {field: _median_of_measured([getattr(d, field) for d in chance]) for field in HandDistance.model_fields}
    )


def _median_of_measured(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return median(measured) if measured else None


def _hand(ink: str, slant: float) -> HandLook:
    return HandLook(slant=slant, stroke=0.07, letters=0.3, density=0.12, paper="#F0E0C0", ink=ink)


def test_carry_over_is_the_slope_of_the_hands_on_their_references() -> None:
    references = [_hand("#202020", 10.0), _hand("#606060", 20.0), _hand("#A0A0A0", 30.0)]
    hands = [_hand(reference.ink, slant) for reference, slant in zip(references, (15.0, 20.0, 25.0), strict=True)]

    carried = carry_over(list(zip(references, hands, strict=True)))

    assert carried.model_dump() == pytest.approx(
        {"paper_lightness": None, "paper_yellowness": None, "ink_lightness": 1.0, "ink_yellowness": None, "slant": 0.5}
    )


def test_read_crops_refuses_a_set_where_only_some_crops_name_a_reference(tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg", reference="ref.jpg"), _crop("b.jpg")])

    with pytest.raises(ValueError, match="1 of 2 crops name a reference image: all or none must"):
        read_crops(crops, 2)
