"""The evaluate CLI end to end on CPU (diffusers' tiny FLUX.3 test model, the real HTR read-back): every crop drawn
and scored into results.json, a crop that fails recorded while the rest still run."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.evaluate import EvalCrop, Results, app, read_crops

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


def _crop(name: str, reference: str | None = None) -> EvalCrop:
    return EvalCrop(
        file_name=name, caption="A handwritten page.", width=64, height=32, lines=LINES, reference=reference
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


def test_read_crops_refuses_a_set_where_only_some_crops_name_a_reference(tmp_path: Path) -> None:
    crops = _eval_set(tmp_path / "crops", [_crop("a.jpg", reference="ref.jpg"), _crop("b.jpg")])

    with pytest.raises(ValueError, match="1 of 2 crops name a reference image: all or none must"):
        read_crops(crops, 2)
