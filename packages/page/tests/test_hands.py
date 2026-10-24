"""Measuring a saved evaluate run's hands without the GPU: each drawing beside its real crop, read back at the
drawing's size as evaluate reads it, and beside its reference."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.hand import hand_look
from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.evaluate import CropResult, Results, Summary
from diffu_page.cli.hands import HandsReport, app
from diffu_page.evalset import EvalCrop

LINES = [
    LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 500, 1000]),
    LineRecord(id="Sv_Text_2", text="den 4 maj", role="", bbox=[500, 0, 1000, 1000]),
]
REFERENCE_LINES = [[100, 50, 300, 950], [500, 50, 700, 950]]


def _run(directory: Path, drawing: Image.Image) -> Path:
    """An evaluate run that drew one crop as ``drawing``."""
    (directory / "generated").mkdir(parents=True)
    drawing.save(directory / "generated" / "000.png")
    summary = Summary(n=1, failed=0, gen_cer=0.5, real_cer=0.1, loras={})
    sample = CropResult(index=0, file="a.jpg", gen_cer=0.5, real_cer=0.1, lines=[])
    (directory / "results.json").write_text(Results(summary=summary, samples=[sample], failed={}).model_dump_json())
    return directory


def test_hands_measures_a_saved_drawing_beside_its_real_crop_and_its_reference(tmp_path: Path) -> None:
    crops = tmp_path / "crops"
    crops.mkdir()
    crop = EvalCrop(
        file_name="a.jpg",
        caption="A handwritten page.",
        width=64,
        height=32,
        lines=LINES,
        reference="ref.png",
        reference_lines=REFERENCE_LINES,
    )
    (crops / "metadata.jsonl").write_text(f"{crop.model_dump_json()}\n")
    Image.linear_gradient("L").resize((96, 48)).convert("RGB").save(crops / "a.jpg")  # larger than the drawing
    reference = Image.linear_gradient("L").rotate(30).convert("RGB")
    reference.save(crops / "ref.png")
    drawing = Image.linear_gradient("L").rotate(90).resize((64, 32)).convert("RGB")
    run = _run(tmp_path / "run", drawing)

    exit_code = CliRunner().invoke(app, ["--crops", str(crops), "--run", str(run)]).exit_code

    hands = HandsReport.model_validate_json((run / "hands.json").read_text()).samples[0].hands
    boxes = [line.bbox for line in LINES]
    with Image.open(crops / "a.jpg") as real:
        real_at_drawing_size = real.convert("RGB").resize(drawing.size)
    assert exit_code == 0
    assert (hands.drawn, hands.real, hands.reference) == (
        hand_look(drawing, boxes),
        hand_look(real_at_drawing_size, boxes),
        hand_look(reference, REFERENCE_LINES),
    )
