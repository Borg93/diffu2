"""The option types the generation CLIs share: a LoRA option names a NeMo LoRA checkpoint dir, checked when the
option is read, before any model loads; the model comes from --model or FLUX3_MODEL, never a default path."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from diffu_page.cli.options import Lora, lora_dirs, parse_lora, required_model
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
