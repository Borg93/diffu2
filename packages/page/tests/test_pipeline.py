"""The page and crop pipelines end to end on CPU: diffusers' tiny FLUX.3 test model, the real HTR read-back, no GPU.

The tiny model draws noise, so every line reads back badly: what is checked is the wiring (a plan in, an image of the
plan's size with one label per line out; a reference image drawn from as it was resized for training, and guided
toward when asked; diffu-generate drawing each planned crop on its own, so one that fails costs no other) and that a
NeMo LoRA loads at its trained strength.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from diffu_page.blocks import DrawnCrop
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.cli.generate import Summary, draw_pages
from diffu_page.cli.generate import app as generate_app
from diffu_page.crops.grid import REF_PIXELS, as_reference
from diffu_page.jobs import PageJob
from diffu_page.lora import ADAPTER_CONFIG, ADAPTER_WEIGHTS, nemo_lora_state_dict
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline, model_family
from diffu_page.plan import CropPlan, PageStyle, Reference

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
    assert _alto_lines(crop.labels_xml) == ["den 4 maj", "Göteborgs poliskammare"]  # in the plan's reading order


def test_crop_pipeline_draws_a_plan_with_the_model_a_diffu_repo_names(tiny_diffu: Path) -> None:
    pipe = load_pipeline(tiny_diffu, crop_blocks, loras={}, settings=TINY_SETTINGS)

    crop = _draw(pipe)

    assert crop.image.size == (64, 32)
    assert _alto_lines(crop.labels_xml) == ["den 4 maj", "Göteborgs poliskammare"]  # in the plan's reading order


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


def test_draw_pages_records_a_crop_it_cannot_draw_by_name_and_saves_the_other_crops(
    crop_pipe: ModularPipeline, two_column_page: Path, wide_and_narrow_body_page: Path, tmp_path: Path
) -> None:
    unreadable = Reference(image=tmp_path / "missing.png")  # planned without it; each of its crops fails to read it
    pages = [
        PageJob(template=two_column_page, text=TEXT, style=STYLE, reference=unreadable),
        PageJob(template=wide_and_narrow_body_page, text=TEXT, style=STYLE),
    ]

    crops, failed = draw_pages(crop_pipe, pages, settings=TINY_SETTINGS, out=tmp_path, prof=None)

    assert [crop.name for crop in crops] == ["001_wide_and_narrow_0"]
    assert {name: error.split(":")[0] for name, error in failed.items()} == {
        "000_two_columns_0": "FileNotFoundError",
        "000_two_columns_1": "FileNotFoundError",
    }
    assert sorted(path.name for path in tmp_path.glob("0*")) == [
        "001_wide_and_narrow_0.jpg",
        "001_wide_and_narrow_0.xml",
    ]


def test_generate_records_what_drew_the_run_and_a_page_that_plans_to_no_crop(
    tiny_flux3: Path, one_line_page: Path, body_and_notes_page: Path, tmp_path: Path
) -> None:
    args = ["--templates", str(one_line_page), "--text-from", str(body_and_notes_page), "--title", "the archive"]
    args += ["--model", str(tiny_flux3), "--out", str(tmp_path), "--steps", "2", "--text-layer", "1", "--device", "cpu"]

    result = CliRunner().invoke(generate_app, args)

    summary = Summary.model_validate_json((tmp_path / "summary.json").read_text())
    assert result.exit_code == 1
    assert (summary.model, summary.seed, summary.settings, summary.crops) == (
        str(tiny_flux3),
        0,
        DrawSettings(steps=2, text_layers=(1,), device="cpu"),
        [],
    )
    assert {name: error.split(":")[0] for name, error in summary.failed.items()} == {"000_one_line": "ValueError"}


@pytest.mark.parametrize(
    ("index", "error"),
    [
        pytest.param(
            {"_blocks_class_name": "OtherAutoBlocks"},
            "names the blocks 'OtherAutoBlocks'; the known model families are",
            id="unknown",
        ),
        pytest.param({}, "names no _blocks_class_name", id="unnamed"),
    ],
)
def test_model_family_refuses_a_repo_whose_blocks_it_does_not_know(
    tmp_path: Path, index: dict[str, str], error: str
) -> None:
    (tmp_path / "modular_model_index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match=error):
        model_family(tmp_path)


def test_draw_crop_refuses_reference_guidance_on_a_crop_without_a_reference(crop_pipe: ModularPipeline) -> None:
    guided = TINY_SETTINGS.model_copy(update={"reference_guidance_scale": 3.0})

    with pytest.raises(ValueError, match="reference guidance needs a reference image"):
        draw_crop(crop_pipe, _plan(), name="000", seed=0, reference=None, settings=guided)


def test_nemo_lora_loads_at_its_trained_strength(
    tiny_flux3: Path, crop_pipe: ModularPipeline, write_nemo_lora: Callable[..., Path]
) -> None:
    torch.manual_seed(0)
    linear = crop_pipe.transformer.get_submodule(LORA_MODULE)
    a = torch.randn(2, linear.in_features)
    b = torch.randn(linear.out_features, 2)
    lora_dir = write_nemo_lora(LORA_MODULE, a, b, alpha=4)  # scale alpha / r = 2
    x = torch.randn(3, linear.in_features)
    expected = linear(x.to(linear.weight.dtype)).float() + 2 * x @ a.T @ b.T

    pipe = load_pipeline(tiny_flux3, crop_blocks, loras={"v2": lora_dir}, settings=TINY_SETTINGS)

    got = pipe.transformer.get_submodule(LORA_MODULE)(x.to(linear.weight.dtype)).float()
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-1)


def _rslora(lora_dir: Path) -> None:
    config = json.loads((lora_dir / ADAPTER_CONFIG).read_text())
    (lora_dir / ADAPTER_CONFIG).write_text(json.dumps({**config, "use_rslora": True}))


def _unprefixed(lora_dir: Path) -> None:
    weights = load_file(lora_dir / ADAPTER_WEIGHTS)
    save_file(
        {key.removeprefix("base_model.model."): value for key, value in weights.items()}, lora_dir / ADAPTER_WEIGHTS
    )


@pytest.mark.parametrize(
    ("spoil", "error"),
    [
        pytest.param(_rslora, "rsLoRA", id="rslora"),
        pytest.param(_unprefixed, "lack NeMo's 'base_model.model.' prefix", id="not-nemo"),
    ],
)
def test_nemo_lora_refuses_a_checkpoint_it_would_load_at_the_wrong_strength_or_place(
    write_nemo_lora: Callable[..., Path], spoil: Callable[[Path], None], error: str
) -> None:
    lora_dir = write_nemo_lora(LORA_MODULE, torch.randn(2, 8), torch.randn(8, 2), alpha=4)
    spoil(lora_dir)

    with pytest.raises(ValueError, match=error):
        nemo_lora_state_dict(lora_dir)
