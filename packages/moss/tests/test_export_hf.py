"""diffu-moss-export-hf on CPU: a MOSS model directory written in the official Hub layout, its DiT's tensors all
reachable through the weights' index beside the rotary frequencies, by link or by copy, and those frequencies the
official checkpoint's. That SGLang draws from the folder as our pipeline does is packages/sglang_moss's test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from diffu_core.repo import ModelIndex
from diffu_moss.export_hf import export_hf, rope_bands
from diffu_moss.train.records import EXPORT_RECORD, ExportRecord

# rope.bands of MOSS-Image-Base's transformer (revision 85975793), float32 values as float.hex
OFFICIAL_BANDS = Path(__file__).parent / "data" / "official_rope_bands.json"


def _tensors(transformer: Path) -> dict[str, torch.Tensor]:
    """Every tensor of a DiT folder, read through its weights' index."""
    index = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))
    tensors = {}
    for name, shard in index["weight_map"].items():
        with safe_open(str(transformer / shard), "pt") as weights:
            tensors[name] = weights.get_tensor(name)
    return tensors


@pytest.mark.parametrize("copy", [False, True], ids=["linked", "copied"])
def test_export_writes_the_official_layout_with_the_dit_and_the_official_rope_bands(
    tiny_moss_reference: Path, tmp_path: Path, copy: bool
) -> None:
    out = tmp_path / "hf"

    takes_reference = export_hf(tiny_moss_reference, out, copy=copy)

    index = json.loads((out / "model_index.json").read_text(encoding="utf-8"))
    config = json.loads((out / "transformer" / "config.json").read_text(encoding="utf-8"))
    assert (index["_class_name"], index["transformer"], config["_class_name"]) == (
        "MOSSImagePipeline",
        ["diffusers", "MOSSImageTransformer"],
        "MOSSImageTransformer",
    )
    assert takes_reference and ExportRecord.read(out / "transformer" / EXPORT_RECORD).takes_reference
    assert all((out / name).is_symlink() != copy for name in ("text_encoder", "tokenizer", "vae", "scheduler"))
    tensors = _tensors(out / "transformer")
    dit = Path(ModelIndex.read(tiny_moss_reference).source("transformer").repo)
    with safe_open(str(dit / "diffusion_pytorch_model.safetensors"), "pt") as weights:
        assert all(torch.equal(tensors[name], weights.get_tensor(name)) for name in weights.keys())  # noqa: SIM118
    assert "rope.bands" in tensors and torch.equal(tensors["rope.bands"], rope_bands(config))


def test_rope_bands_at_moss_images_width_are_the_official_checkpoints() -> None:
    official = json.loads(OFFICIAL_BANDS.read_text(encoding="utf-8"))

    bands = rope_bands({"dim": 3072, "num_heads": 24, "rope_base": 10000.0})

    assert torch.equal(bands, torch.tensor([float.fromhex(value) for value in official["hex"]], dtype=torch.float32))
