"""The evaluate CLI end to end on CPU (diffusers' tiny FLUX.3 test model, the real HTR read-back): every crop drawn
and scored into results.json, its hand measured beside its real crop's and its reference's, a crop that fails
recorded while the rest still run."""

from __future__ import annotations

from pathlib import Path
from statistics import median

import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner, Result

from diffu_page.captions.hand import HandDistance, HandLook, hand_distance, hand_look
from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.evaluate import Results, Summary, app
from diffu_page.evalset import EvalCrop, read_crops
from diffu_page.fidelity import Hands, carry_over
from diffu_page.pipeline import DrawSettings

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


def _evaluate(tiny_flux3: Path, crops: Path, out: Path, *options: str) -> Result:
    args = ["--crops", str(crops), "--out", str(out), "--model", str(tiny_flux3), "-n", "2", "--steps", "2"]
    return CliRunner().invoke(app, [*args, "--text-layer", "1", "--device", "cpu", *options])


def _referenced_set(crops_dir: Path) -> tuple[Path, list[Image.Image]]:
    """Two crops, each with its own reference image (gradients at two angles) and the reference's line boxes."""
    crops = _eval_set(
        crops_dir, [_crop("a.jpg", "ref_a.png", REFERENCE_LINES), _crop("b.jpg", "ref_b.png", REFERENCE_LINES)]
    )
    references = [Image.linear_gradient("L").rotate(angle).convert("RGB") for angle in (30, 120)]
    for name, reference in zip(("ref_a.png", "ref_b.png"), references, strict=True):
        reference.save(crops / name)
    return crops, references


def test_evaluate_records_a_failed_crop_and_scores_the_rest(tiny_flux3: Path, tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg"), _crop("b.jpg")])
    (crops / "b.jpg").unlink()  # its real crop can't be read back

    result = _evaluate(tiny_flux3, crops, tmp_path / "out")

    results = Results.model_validate_json((tmp_path / "out" / "results.json").read_text())
    assert result.exit_code == 1
    assert [(sample.index, sample.file) for sample in results.samples] == [(0, "a.jpg")]
    assert list(results.failed) == [1]
    last = [line for line in result.output.splitlines() if line.strip()][-1]
    assert Summary.model_validate_json(last) == results.summary  # one line, for the job logs


def test_evaluate_measures_each_drawn_hand_beside_its_real_crop_and_its_reference(
    tiny_flux3: Path, tmp_path: Path
) -> None:
    crops, references = _referenced_set(tmp_path / "crops")

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "out").exit_code

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


def test_evaluate_guides_toward_each_crops_reference_when_asked(tiny_flux3: Path, tmp_path: Path) -> None:
    crops, _ = _referenced_set(tmp_path / "crops")
    _evaluate(tiny_flux3, crops, tmp_path / "caption")

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "reference", "--reference-guidance", "3").exit_code

    results = Results.model_validate_json((tmp_path / "reference" / "results.json").read_text())
    drawn = [(tmp_path / run / "generated" / "000.png").read_bytes() for run in ("caption", "reference")]
    assert exit_code == 0
    assert results.summary.settings == DrawSettings(
        steps=2, reference_guidance_scale=3.0, text_layers=(1,), device="cpu"
    )
    assert drawn[1] != drawn[0]


def test_evaluate_without_reference_draws_each_crop_from_its_caption_alone_and_measures_it_beside_its_reference(
    tiny_flux3: Path, tmp_path: Path
) -> None:
    crops, references = _referenced_set(tmp_path / "crops")
    _evaluate(tiny_flux3, _eval_set(tmp_path / "plain", [_crop("a.jpg")]), tmp_path / "caption", "-n", "1")

    exit_code = _evaluate(tiny_flux3, crops, tmp_path / "out", "-n", "1", "--without-reference").exit_code

    results = Results.model_validate_json((tmp_path / "out" / "results.json").read_text())
    drawn = [(tmp_path / run / "generated" / "000.png").read_bytes() for run in ("caption", "out")]
    assert exit_code == 0
    assert drawn[1] == drawn[0]  # as from the same caption with no reference at all
    assert results.samples[0].hands.reference == hand_look(references[0], REFERENCE_LINES)


def test_evaluate_refuses_reference_guidance_on_crops_without_references_before_it_loads(tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg"), _crop("b.jpg")])

    result = CliRunner().invoke(
        app,
        ["--crops", str(crops), "--out", str(tmp_path / "out"), "--model", str(tmp_path / "no_model")]
        + ["--reference-guidance", "3"],
    )

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2  # a usage error, raised before the (missing) model would have been read
    assert "these crops name no reference image to guide toward" in message


def _median_of_measured(values: list[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return median(measured) if measured else None


def _hand(ink: str | None, slant: float | None) -> HandLook:
    return HandLook(slant=slant, stroke=0.07, letters=0.3, density=0.12, paper="#F0E0C0", ink=ink)


def test_carry_over_is_the_slope_of_the_measured_hands_on_their_references() -> None:
    references = [_hand("#202020", 10.0), _hand("#606060", 20.0), _hand("#A0A0A0", 30.0), _hand("#E0E0E0", 40.0)]
    hands = [_hand(reference.ink, slant) for reference, slant in zip(references, (15.0, 20.0, 25.0), strict=False)]
    blank = _hand(None, None)  # a drawing with no ink: left out, not read as black ink

    carried = carry_over(list(zip(references, [*hands, blank], strict=True)))

    assert carried.model_dump() == pytest.approx(
        {"paper_lightness": None, "paper_yellowness": None, "ink_lightness": 1.0, "ink_yellowness": None, "slant": 0.5}
    )


def test_eval_crop_refuses_a_reference_without_the_line_boxes_that_measure_its_hand() -> None:
    with pytest.raises(ValidationError, match="a reference and its line boxes go together"):
        _crop("a.jpg", reference="ref.png")


def test_read_crops_refuses_a_set_where_only_some_crops_name_a_reference(tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg", "ref.jpg", REFERENCE_LINES), _crop("b.jpg")])

    with pytest.raises(ValueError, match="1 of 2 crops name a reference image: all or none must"):
        read_crops(crops, 2)
