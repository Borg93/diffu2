"""The inference benchmark (python -m diffu_moss.bench_inference) on CPU with conftest's tiny MOSS: one line per
configuration asked, in the order asked; the first the reference (its cosines 1), MXFP8's close to it but not equal,
each step timed and each configuration's page saved; a configuration that fails (the tiny text encoder is too narrow
for MXFP8) gets a line of its error and the next one still runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from diffu_moss.bench_inference import app


def test_bench_inference_records_each_configuration_against_the_first(tiny_moss: Path, tmp_path: Path) -> None:
    options = ["--model", str(tiny_moss), "--device", "cpu", "--size", "64x32", "--steps", "2"]
    options += ["--config", "bf16_bf16_eager", "--config", "mxfp8_mxfp8_eager", "--config", "mxfp8_bf16_eager"]

    result = CliRunner().invoke(app, [*options, "--out", str(tmp_path / "bench.jsonl"), "--images", str(tmp_path)])

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in (tmp_path / "bench.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [record["config"] for record in records] == ["bf16_bf16_eager", "mxfp8_mxfp8_eager", "mxfp8_bf16_eager"]
    reference, failed, mxfp8 = records
    assert failed == {
        "config": "mxfp8_mxfp8_eager",
        "error": "ValueError: MXFP8 takes layers of a multiple of 32 inputs and 16 outputs; 11 chosen layers are not, "
        "the first language_model.layers.0.linear_attn.out_proj",
    }
    assert (reference["velocity_cosine"], reference["latents_cosine"]) == (pytest.approx(1.0), pytest.approx(1.0))
    assert 0.99 < mxfp8["velocity_cosine"] < 1 - 1e-9
    assert 0.99 < mxfp8["latents_cosine"] < 1 - 1e-9
    assert [len(record["step_seconds_each"]) for record in (reference, mxfp8)] == [2, 2]
    for record in (reference, mxfp8):
        with Image.open(tmp_path / f"{record['config']}.png") as page:
            assert page.size == (64, 32)
