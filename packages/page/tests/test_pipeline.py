"""The page and crop pipelines end to end on CPU: diffusers' tiny FLUX.3 test model, the real HTR read-back, no GPU.

The tiny model draws noise, so every line reads back badly: what is checked is the wiring (a plan in, an image of the
plan's size with one label per line out; a reference image drawn from as it was resized for training, and guided
toward when asked; the page pipeline drawing every planned crop exactly as the crop pipeline draws it, from the page's
reference when it has one) and that a NeMo LoRA loads at its trained strength.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image

from diffu_page.blocks import DrawnCrop
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.crops.grid import REF_PIXELS, as_reference
from diffu_page.layout import load_xml_layout, refill
from diffu_page.lora import load_nemo_lora
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline, model_family, page_blocks
from diffu_page.plan import CropPlan, PageStyle, Reference, plan_page

TINY_SETTINGS = DrawSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
LORA_MODULE = "single_transformer_blocks.0.attn.to_q"
STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4


@pytest.fixture(scope="module")
def crop_pipe(tiny_flux3: Path) -> ModularPipeline:
    return load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=TINY_SETTINGS)


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


def _draw(pipe: ModularPipeline, reference: Image.Image | None = None) -> DrawnCrop:
    return draw_crop(pipe, _plan(), name="000", seed=0, reference=reference, settings=TINY_SETTINGS)


def _alto_lines(labels_xml: str) -> list[str]:
    root = ET.fromstring(labels_xml)
    lines = [e for e in root.iter() if e.tag.endswith("TextLine")]
    return [" ".join(s.get("CONTENT", "") for s in line.iter() if s.tag.endswith("String")) for line in lines]


def test_crop_pipeline_draws_a_plan_with_one_unclear_label_per_line(crop_pipe: ModularPipeline) -> None:
    crop = _draw(crop_pipe)

    assert crop.image.size == (64, 32)
    assert len(crop.line_cers) == 2
    assert crop.unclear_lines == 2  # noise reads back as nothing like the text
    assert sorted(_alto_lines(crop.labels_xml)) == ["Göteborgs poliskammare", "den 4 maj"]


def test_crop_pipeline_draws_a_plan_with_the_model_a_diffu_repo_names(tiny_diffu: Path) -> None:
    pipe = load_pipeline(tiny_diffu, crop_blocks, loras={}, settings=TINY_SETTINGS)

    crop = _draw(pipe)

    assert crop.image.size == (64, 32)
    assert sorted(_alto_lines(crop.labels_xml)) == ["Göteborgs poliskammare", "den 4 maj"]


def test_crop_pipeline_conditions_a_crop_on_its_reference_image(crop_pipe: ModularPipeline) -> None:
    plain = _draw(crop_pipe)

    crop = _draw(crop_pipe, Image.new("RGB", (48, 32), "white"))

    assert crop.image.size == (64, 32)
    assert crop.image.tobytes() != plain.image.tobytes()


def test_crop_pipeline_guided_toward_its_reference_draws_otherwise_than_guided_by_the_caption_alone(
    tiny_flux3: Path, crop_pipe: ModularPipeline
) -> None:
    reference = Image.new("RGB", (48, 32), "white")
    settings = TINY_SETTINGS.model_copy(update={"reference_guidance_scale": 3.0})
    guided_pipe = load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=settings)
    caption_guided = _draw(crop_pipe, reference)

    reference_guided = draw_crop(guided_pipe, _plan(), name="000", seed=0, reference=reference, settings=settings)

    assert reference_guided.image.tobytes() != caption_guided.image.tobytes()


def _off_grid_reference() -> Image.Image:
    return Image.linear_gradient("L").resize((50, 34)).convert("RGB")


def test_as_reference_squashes_a_small_reference_onto_the_16_px_grid() -> None:
    reference = _off_grid_reference()

    gridded = as_reference(reference)

    assert gridded.size == (48, 32)
    assert gridded.tobytes() == reference.resize((48, 32), Image.Resampling.LANCZOS).tobytes()


def test_as_reference_shrinks_a_full_crop_to_the_size_references_were_trained_at() -> None:
    crop = Image.linear_gradient("L").resize((1472, 640)).convert("RGB")  # a 0.94 MP training-bucket crop

    reference = as_reference(crop)

    assert reference.size == (1088, 464)
    assert reference.width * reference.height <= REF_PIXELS


def test_as_reference_leaves_a_reference_sized_image_unchanged() -> None:
    reference = as_reference(Image.linear_gradient("L").resize((1472, 640)).convert("RGB"))

    again = as_reference(reference)

    assert (again.size, again.tobytes()) == (reference.size, reference.tobytes())


def test_crop_pipeline_draws_from_a_reference_as_resized_for_training(crop_pipe: ModularPipeline) -> None:
    reference = _off_grid_reference()
    as_trained = _draw(crop_pipe, as_reference(reference))

    crop = _draw(crop_pipe, reference)

    assert crop.image.tobytes() == as_trained.image.tobytes()


def _identity(crop: DrawnCrop) -> tuple[str, CropPlan, str, str]:
    """A crop's name, plan, image digest and labels: what must match between two draws of it."""
    return crop.name, crop.plan, hashlib.sha256(crop.image.tobytes()).hexdigest(), crop.labels_xml


@pytest.mark.parametrize("drawn_from_reference", [pytest.param(False, id="style"), pytest.param(True, id="reference")])
def test_page_pipeline_draws_every_crop_as_the_crop_pipeline_draws_it(
    tiny_flux3: Path, crop_pipe: ModularPipeline, two_column_page: Path, tmp_path: Path, drawn_from_reference: bool
) -> None:
    reference = None
    if drawn_from_reference:
        _off_grid_reference().save(tmp_path / "reference.png")
        reference = Reference(image=tmp_path / "reference.png")
    page_pipe = load_pipeline(tiny_flux3, page_blocks, loras={}, settings=TINY_SETTINGS)
    plans = plan_page(refill(load_xml_layout(two_column_page), [TEXT]), STYLE, reference=reference)
    image = _off_grid_reference() if drawn_from_reference else None
    expected = [
        draw_crop(crop_pipe, plan, name=f"p_{k}", seed=7 + k, reference=image, settings=TINY_SETTINGS)
        for k, plan in enumerate(plans)
    ]

    crops: list[DrawnCrop] = page_pipe(
        template=two_column_page,
        text=TEXT,
        style=STYLE,
        reference=reference,
        page_name="p",
        seed=7,
        **TINY_SETTINGS.pipeline_inputs(),
        output="crops",
    )

    assert [_identity(crop) for crop in crops] == [_identity(crop) for crop in expected]


def test_model_family_refuses_a_repo_whose_blocks_it_does_not_know(tmp_path: Path) -> None:
    (tmp_path / "modular_model_index.json").write_text(json.dumps({"_blocks_class_name": "OtherAutoBlocks"}))

    with pytest.raises(ValueError, match="names the blocks 'OtherAutoBlocks'; the known model families are"):
        model_family(tmp_path)


def test_nemo_lora_loads_at_its_trained_strength(tiny_flux3: Path, write_nemo_lora: Callable[..., Path]) -> None:
    torch.manual_seed(0)
    pipe = load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=TINY_SETTINGS)
    linear = pipe.transformer.get_submodule(LORA_MODULE)
    a = torch.randn(2, linear.in_features)
    b = torch.randn(linear.out_features, 2)
    lora_dir = write_nemo_lora(LORA_MODULE, a, b, alpha=4)  # scale alpha / r = 2
    x = torch.randn(3, linear.in_features)
    expected = linear(x.to(linear.weight.dtype)).float() + 2 * x @ a.T @ b.T

    load_nemo_lora(pipe, lora_dir, "v2")

    got = pipe.transformer.get_submodule(LORA_MODULE)(x.to(linear.weight.dtype)).float()
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-1)
