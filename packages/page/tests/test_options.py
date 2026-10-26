"""The option types the generation CLIs share: a LoRA option names a NeMo LoRA checkpoint dir, checked when the
option is read, before any model loads; the model comes from --model or FLUX3_MODEL, never a default path; every draw
option a CLI is given reaches the settings it draws with."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from diffu_page.cli.generate import Summary
from diffu_page.cli.generate import app as generate_app
from diffu_page.cli.options import Lora, lora_dirs, parse_lora, required_model
from diffu_page.labels import LabelFormat
from diffu_page.pipeline import DrawSettings
from diffu_page.settings import Settings


def test_parse_lora_refuses_a_dir_that_holds_no_nemo_lora(tmp_path: Path) -> None:
    with pytest.raises(typer.BadParameter, match="is not a NeMo LoRA model dir: it has no adapter_config.json"):
        parse_lora(f"hand={tmp_path}")


def test_a_command_given_neither_flux3_model_nor_model_asks_for_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FLUX3_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)  # and no .env file

    with pytest.raises(typer.BadParameter, match="set FLUX3_MODEL"):
        required_model(Settings().flux3_model)


def test_lora_dirs_refuse_two_loras_of_one_name(tmp_path: Path) -> None:
    loras = [Lora(name="v2", path=tmp_path / "a"), Lora(name="v2", path=tmp_path / "b")]

    with pytest.raises(typer.BadParameter, match="two LoRAs are named 'v2'"):
        lora_dirs(loras)


def test_generate_sets_each_draw_setting_from_its_option(
    one_line_page: Path, body_and_notes_page: Path, tmp_path: Path
) -> None:
    args = ["--templates", str(one_line_page), "--text-from", str(body_and_notes_page), "--title", "the archive"]
    args += ["--model", str(tmp_path / "no_model"), "--out", str(tmp_path)]
    args += ["--reference", str(tmp_path / "ref.png"), "--reference-guidance", "1.5"]  # guidance needs a reference
    args += ["--steps", "3", "--guidance", "2.5", "--guidance-stop", "0.5", "--text-layer", "1", "--text-layer", "2"]
    args += ["--device", "cpu", "--max-line-cer", "0.1", "--format", "page", "--lora-scale", "0.75"]
    args += ["--sampling-shift", "3", "--fuse-loras", "--compile-blocks", "--repair", "2"]

    CliRunner().invoke(generate_app, args)  # the page plans to no crop, so no model loads

    summary = Summary.model_validate_json((tmp_path / "summary.json").read_text())
    assert summary.settings == DrawSettings(
        steps=3,
        guidance_scale=2.5,
        reference_guidance_scale=1.5,
        guidance_stop=0.5,
        label_format=LabelFormat.PAGE,
        max_line_cer=0.1,
        text_layers=(1, 2),
        device="cpu",
        lora_scale=0.75,
        sampling_shift=3.0,
        fuse_loras=True,
        compile_blocks=True,
        repair_passes=2,
    )
