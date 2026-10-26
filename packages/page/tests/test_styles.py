"""Style presets read from a presets file: a preset whose LoRA is not a NeMo LoRA checkpoint is refused when the file
is read, as ``--lora`` refuses one, naming the file and the preset. How a job file's named styles resolve into pages,
LoRAs and references is test_pipeline's: diffu-generate reads both files."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from diffu_page.captions.prompt import PageStyle
from diffu_page.styles import read_style_presets

if TYPE_CHECKING:
    from conftest import NemoLoraWriter


def test_read_style_presets_names_the_file_and_the_preset_whose_lora_is_no_nemo_lora(
    tmp_path: Path, page_style: PageStyle, write_nemo_lora: NemoLoraWriter
) -> None:
    hand, not_a_lora = write_nemo_lora("to_q", torch.zeros(2, 8), torch.zeros(8, 2), alpha=2), tmp_path / "b_lora"
    presets = tmp_path / "styles.json"
    style = page_style.model_dump(mode="json")
    presets.write_text(
        json.dumps({"a": {**style, "lora": str(hand)}, "b": {**style, "lora": str(not_a_lora)}}), encoding="utf-8"
    )
    problem = f"b.lora: Value error, {not_a_lora} is not a NeMo LoRA model dir: it has no adapter_config.json"

    with pytest.raises(ValueError, match=re.escape(f"{presets} is not a file of style presets: {problem}")):
        read_style_presets(presets)
