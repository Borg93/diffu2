"""The crop generator end to end on CPU: diffusers' tiny FLUX.3 test model, the real HTR read-back, no GPU.

The tiny model draws noise, so every line reads back badly: what is checked is the wiring (a plan in, an image of the
plan's size with one label per line out, with and without a reference image) and that a NeMo LoRA loads at its trained
strength.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import save_file

from page_synth.caption import LineRecord
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.plan import CropPlan

TINY_FLUX3 = "hf-internal-testing/tiny-flux3-image-modular"
TINY_SETTINGS = GeneratorSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
LORA_MODULE = "single_transformer_blocks.0.attn.to_q"


@pytest.fixture(scope="module")
def tiny_model() -> Path:
    return Path(snapshot_download(TINY_FLUX3))


@pytest.fixture(scope="module")
def generator(tiny_model: Path) -> CropGenerator:
    return CropGenerator(tiny_model, loras={}, settings=TINY_SETTINGS)


def _plan() -> CropPlan:
    return CropPlan(
        rect=(0, 0, 64, 32),
        width=64,
        height=32,
        caption="A handwritten page. [{...}]",
        lines=[
            LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 500, 1000]),
            LineRecord(id="Sv_Text_2", text="den 4 maj", role="marginalia", bbox=[500, 0, 1000, 400]),
        ],
        reading_order=[1, 0],
    )


def _alto_lines(labels_xml: str) -> list[str]:
    root = ET.fromstring(labels_xml)
    lines = [e for e in root.iter() if e.tag.endswith("TextLine")]
    return [" ".join(s.get("CONTENT") for s in line.iter() if s.tag.endswith("String")) for line in lines]


def test_generator_draws_a_plan_with_one_unclear_label_per_line(generator: CropGenerator) -> None:
    crop = generator(_plan(), seed=0, image_name="000.jpg")

    assert crop.image.size == (64, 32)
    assert len(crop.line_cers) == 2
    assert crop.unclear_lines == 2  # noise reads back as nothing like the text
    assert sorted(_alto_lines(crop.labels_xml)) == ["Göteborgs poliskammare", "den 4 maj"]


def test_generator_conditions_a_crop_on_its_reference_image(generator: CropGenerator) -> None:
    plain = generator(_plan(), seed=0, image_name="000.jpg")
    reference = Image.new("RGB", (48, 32), "white")

    crop = generator(_plan(), seed=0, image_name="000.jpg", reference=reference)

    assert crop.image.size == (64, 32)
    assert crop.image.tobytes() != plain.image.tobytes()


def _write_nemo_lora(lora_dir: Path, a: torch.Tensor, b: torch.Tensor, *, alpha: int) -> None:
    lora_dir.mkdir()
    save_file(
        {
            f"base_model.model.{LORA_MODULE}.lora_A.weight": a,
            f"base_model.model.{LORA_MODULE}.lora_B.weight": b,
        },
        lora_dir / "adapter_model.safetensors",
    )
    config = {"r": a.shape[0], "lora_alpha": alpha, "target_modules": [LORA_MODULE], "peft_type": "LORA"}
    (lora_dir / "adapter_config.json").write_text(json.dumps(config))


def test_nemo_lora_loads_at_its_trained_strength(tiny_model: Path, tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = CropGenerator(tiny_model, loras={}, settings=TINY_SETTINGS)._pipes.crop.transformer
    linear = base.get_submodule(LORA_MODULE)
    a = torch.randn(2, linear.in_features)
    b = torch.randn(linear.out_features, 2)
    _write_nemo_lora(tmp_path / "lora", a, b, alpha=4)  # scale alpha / r = 2
    x = torch.randn(3, linear.in_features)
    expected = linear(x.to(linear.weight.dtype)).float() + 2 * x @ a.T @ b.T

    adapted = CropGenerator(tiny_model, loras={"v2": tmp_path / "lora"}, settings=TINY_SETTINGS)
    got = adapted._pipes.crop.transformer.get_submodule(LORA_MODULE)(x.to(linear.weight.dtype)).float()

    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-1)
