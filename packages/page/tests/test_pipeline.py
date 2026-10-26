"""The page and crop pipelines end to end on CPU: diffusers' tiny FLUX.3 test model, the real HTR read-back, no GPU.

The tiny model draws noise, so every line reads back badly: what is checked is the wiring (a plan in, an image of the
plan's size with one label per line out; a reference image drawn from as it was resized for training, and guided
toward when asked; diffu-generate drawing each planned crop on its own, so one that fails costs no other) and that a
NeMo LoRA loads at its trained strength.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
import torch
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from diffu_page.blocks import CropPromptStep, DrawnCrop, ReferenceGridStep
from diffu_page.captions.look import Ink, Paper, ScanLook
from diffu_page.captions.prompt import LineRecord, PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.cli.generate import Summary, draw_pages, plan_jobs
from diffu_page.cli.generate import app as generate_app
from diffu_page.crops.grid import as_reference
from diffu_page.jobs import PageJob
from diffu_page.judge import LineVerdict
from diffu_page.local_run import NOT_RUN
from diffu_page.lora import ADAPTER_CONFIG, ADAPTER_WEIGHTS, nemo_lora_state_dict
from diffu_page.pipeline import (
    FAMILIES,
    CropBlocks,
    DrawSettings,
    ModelFamily,
    crop_blocks,
    draw_crop,
    load_pipeline,
    model_family,
    repair_lines,
)
from diffu_page.plan import CropPlan, Reference

TINY_SETTINGS = DrawSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
LORA_MODULE = "single_transformer_blocks.0.attn.to_q"
STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
TEXT = "Anno sextonhundra och tjugo den femte dagen i maj kom for ratten bonden Olof Jonsson ifran Rodon " * 4
FLAG_STYLE = PageStyle(  # diffu-generate --title "the archive" with every other style flag left at its default
    provenance=Provenance(title="the archive", years=""),
    look=ScanLook(greyscale=False, paper=Paper.CREAM, ink=Ink.DARK_BROWN),
)


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


def test_crop_pipeline_tags_a_line_unclear_above_the_settings_max_line_cer_or_when_its_number_misreads(
    crop_pipe: ModularPipeline,
) -> None:
    lenient = TINY_SETTINGS.model_copy(update={"max_line_cer": 10.0})  # noise reads back at a CER near 1

    crop = draw_crop(crop_pipe, _plan(), name="000", seed=0, reference=None, settings=lenient)

    assert [verdict.passes(10.0) for verdict in crop.verdicts] == [True, False]  # noise reads no "4" in "den 4 maj"
    assert crop.unclear_lines == 1


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


def _off_grid_reference() -> Image.Image:
    return Image.linear_gradient("L").resize((50, 34)).convert("RGB")


def test_crop_pipeline_draws_from_a_reference_as_resized_for_training(crop_pipe: ModularPipeline) -> None:
    reference = _off_grid_reference()
    as_trained = _draw(crop_pipe, as_reference(reference))

    crop = _draw(crop_pipe, reference)

    assert crop.image.tobytes() == as_trained.image.tobytes()


def test_crop_blocks_leave_the_model_familys_own_blocks_as_they_are() -> None:
    family = FAMILIES["Flux3AutoBlocks"]

    crop_blocks(family)

    own = family.blocks().sub_blocks["denoise"]
    assert [list(branch.sub_blocks) for branch in own.sub_blocks.values()] == [
        ["input", "cond_input", "prepare_latents", "prepare_image_cond_latents", "set_timesteps"]
        + ["prepare_rope_inputs", "denoise", "after_denoise"],
        ["input", "prepare_latents", "set_timesteps", "prepare_rope_inputs", "denoise", "after_denoise"],
    ]


def _model_blocks_only(family: ModelFamily) -> CropBlocks:
    """Our crop steps before the model's own blocks, as they stand: no line repair."""
    model = family.blocks().sub_blocks
    for name in family.unused:
        model.pop(name)
    return CropBlocks.from_blocks_dict(
        {"crop_prompt": CropPromptStep(), "reference_grid": ReferenceGridStep(), **model}
    )


def test_crop_pipeline_draws_as_the_models_own_blocks_without_a_repair_mask(
    crop_pipe: ModularPipeline, tiny_flux3: Path
) -> None:
    own = load_pipeline(tiny_flux3, _model_blocks_only, loras={}, settings=TINY_SETTINGS)

    crop = _draw(crop_pipe)

    assert crop.image.tobytes() == _draw(own).image.tobytes()


def _latents(pipe: ModularPipeline, seed: int, **repair: torch.Tensor) -> torch.Tensor:
    pipe.guider.set_state(step=0, num_inference_steps=TINY_SETTINGS.steps, timestep=None)
    return pipe(
        crop_plan=_plan(),
        generator=torch.Generator().manual_seed(seed),
        **TINY_SETTINGS.pipeline_inputs(),
        **repair,
        output="latents",
    )


def test_repair_ends_the_latents_outside_the_mask_on_the_drawn_crops_and_draws_inside_again(
    crop_pipe: ModularPipeline,
) -> None:
    drawn = _latents(crop_pipe, seed=0)
    mask = torch.zeros(1, 1, *drawn.shape[-2:])
    mask[..., :2] = 1.0  # the crop's left half, two of its four latent columns

    redrawn = _latents(crop_pipe, seed=1, repair_latents=drawn, repair_mask=mask)

    assert torch.equal(redrawn[..., 2:], drawn[..., 2:])
    assert not torch.equal(redrawn[..., :2], drawn[..., :2])


class ScriptedJudge:
    """Reads every line's text right, except the lines whose text it is told to misread: those it reads as nothing."""

    def __init__(self, misread: frozenset[str] = frozenset()) -> None:
        self.misread = misread

    def __call__(self, image: Image.Image, lines: Sequence[LineRecord]) -> list[LineVerdict]:
        return [LineVerdict.of("" if line.text in self.misread else line.text, line.text) for line in lines]


def _misdrawn(pipe: ModularPipeline) -> DrawnCrop:
    """The plan drawn with its marginal note "den 4 maj" (rows 500-1000, columns 0-400) rejected."""
    judge = ScriptedJudge(frozenset({"den 4 maj"}))
    return draw_crop(pipe, _plan(), name="000", seed=0, reference=None, settings=TINY_SETTINGS, judge=judge)


def test_repair_lines_puts_back_the_cells_of_a_redrawn_line_the_judge_passes_and_keeps_every_other_pixel(
    crop_pipe: ModularPipeline,
) -> None:
    crop = _misdrawn(crop_pipe)
    cells = (0, 16, 32, 32)  # the 16-px latent cells the note's box (0, 16, 25.6, 32) touches on the 64 x 32 crop
    past_the_box = (26, 16, 32, 32)  # where those cells reach past the box: the redraw meets the crop at cell edges

    repaired = repair_lines(crop_pipe, crop, seed=1, reference=None, settings=TINY_SETTINGS, judge=ScriptedJudge())

    outside = repaired.image.copy()
    outside.paste(crop.image.crop(cells), cells[:2])
    assert outside.tobytes() == crop.image.tobytes()
    assert repaired.image.crop(past_the_box).tobytes() != crop.image.crop(past_the_box).tobytes()
    assert (crop.unclear_lines, repaired.unclear_lines) == (1, 0)


def test_repair_lines_keeps_the_crop_when_the_judge_rejects_the_redrawn_line_too(crop_pipe: ModularPipeline) -> None:
    crop = _misdrawn(crop_pipe)
    judge = ScriptedJudge(frozenset({"den 4 maj"}))

    repaired = repair_lines(crop_pipe, crop, seed=1, reference=None, settings=TINY_SETTINGS, judge=judge)

    assert repaired is crop


def test_draw_pages_records_a_crop_it_cannot_draw_by_name_and_saves_the_other_crops(
    crop_pipe: ModularPipeline, two_column_page: Path, wide_and_narrow_body_page: Path, tmp_path: Path
) -> None:
    unreadable = Reference(image=tmp_path / "missing.png")  # planned without it; each of its crops fails to read it
    pages = [
        PageJob(template=two_column_page, text=TEXT, style=STYLE, reference=unreadable),
        PageJob(template=wide_and_narrow_body_page, text=TEXT, style=STYLE),
    ]

    jobs, unplanned = plan_jobs(pages)

    crops, failed = draw_pages(crop_pipe, jobs, settings=TINY_SETTINGS, out=tmp_path, prof=None)

    assert unplanned == {}
    assert [crop.name for crop in crops] == ["001_wide_and_narrow_0"]
    assert {name: error.split(":")[0] for name, error in failed.items()} == {
        "000_two_columns_0": "FileNotFoundError",
        "000_two_columns_1": "FileNotFoundError",
    }
    assert sorted(path.name for path in tmp_path.glob("0*")) == [
        "001_wide_and_narrow_0.jpg",
        "001_wide_and_narrow_0.xml",
    ]


def test_draw_pages_stops_when_the_device_fails_and_records_the_crops_left_undrawn(
    crop_pipe: ModularPipeline, two_column_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs, _ = plan_jobs([PageJob(template=two_column_page, text=TEXT, style=STYLE)])

    def out_of_memory(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(crop_pipe.transformer, "forward", out_of_memory)

    crops, failed = draw_pages(crop_pipe, jobs, settings=TINY_SETTINGS, out=tmp_path, prof=None)

    assert (crops, failed) == (
        [],
        {
            "000_two_columns_0": "OutOfMemoryError: CUDA out of memory",
            "000_two_columns_1": NOT_RUN,
        },
    )


@pytest.mark.parametrize(
    ("option", "refusal"),
    [
        pytest.param(["--reference-guidance", "3"], "there is no --reference to guide toward", id="guidance"),
        pytest.param(["--reference-caption", "described"], "there is no --reference to describe", id="caption"),
    ],
)
def test_generate_refuses_a_reference_option_without_a_reference_before_it_loads(
    one_line_page: Path, body_and_notes_page: Path, tmp_path: Path, option: list[str], refusal: str
) -> None:
    args = ["--templates", str(one_line_page), "--text-from", str(body_and_notes_page), "--title", "the archive"]
    args += ["--model", str(tmp_path / "no_model"), "--out", str(tmp_path / "out"), *option]

    result = CliRunner().invoke(generate_app, args)

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2  # a usage error, raised before the (missing) model would have been read
    assert refusal in message


def test_generate_records_a_page_that_plans_to_no_crop_without_loading_the_model(
    one_line_page: Path, body_and_notes_page: Path, tmp_path: Path
) -> None:
    no_model = tmp_path / "no_model"  # never read: no page plans to a crop
    args = ["--templates", str(one_line_page), "--text-from", str(body_and_notes_page), "--title", "the archive"]
    args += ["--model", str(no_model), "--out", str(tmp_path), "--steps", "2", "--text-layer", "1", "--device", "cpu"]

    result = CliRunner().invoke(generate_app, args)

    summary = Summary.model_validate_json((tmp_path / "summary.json").read_text())
    assert result.exit_code == 1
    assert (summary.model, summary.settings, summary.crops) == (
        str(no_model),
        DrawSettings(steps=2, text_layers=(1,), device="cpu"),
        [],
    )
    assert [(page.template, page.style, page.seed) for page in summary.pages] == [
        (one_line_page, FLAG_STYLE, 0)  # the style the flags name, with their defaults
    ]
    assert summary.failed == {
        "000_one_line": f"ValueError: {one_line_page} plans to no crops: no window of it holds enough whole lines"
    }


def test_generate_draws_each_page_of_a_job_file_with_its_own_text_style_and_name(
    one_line_page: Path, tmp_path: Path
) -> None:
    pages = [
        PageJob(template=one_line_page, text=TEXT, style=STYLE, id="vol7_p12"),
        PageJob(template=one_line_page, text="den 4 maj", style=FLAG_STYLE, seed=3),
    ]
    job_file = tmp_path / "run.jsonl"
    job_file.write_text("".join(page.model_dump_json() + "\n" for page in pages), encoding="utf-8")
    args = ["--jobs", str(job_file), "--model", str(tmp_path / "no_model"), "--out", str(tmp_path / "out")]

    result = CliRunner().invoke(generate_app, args)

    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    assert result.exit_code == 1  # neither page plans to a crop, so no model loads
    assert summary.pages == pages
    no_crops = f"ValueError: {one_line_page} plans to no crops: no window of it holds enough whole lines"
    assert summary.failed == {"vol7_p12": no_crops, "001_one_line": no_crops}


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        pytest.param(
            ["--jobs", "run.jsonl", "--title", "an archive"],
            "--title: the job file gives each page its own",
            id="jobs-and-flags",
        ),
        pytest.param([], "give --jobs, or --templates, --text-from and --title", id="no-pages"),
    ],
)
def test_generate_refuses_a_run_whose_pages_are_not_given_once_before_it_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], refusal: str
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run.jsonl").write_text("", encoding="utf-8")
    args = ["--model", str(tmp_path / "no_model"), "--out", str(tmp_path / "out"), *options]

    result = CliRunner().invoke(generate_app, args)

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2
    assert refusal in message


def test_generate_writes_the_text_of_a_plain_text_file_as_it_stands(one_line_page: Path, tmp_path: Path) -> None:
    text_file = tmp_path / "text.txt"
    text_file.write_text("Anno 1720 den 5 maj\n\nNästa stycke\n", encoding="utf-8")
    args = ["--templates", str(one_line_page), "--text-from", str(text_file), "--title", "the archive"]
    args += ["--model", str(tmp_path / "no_model"), "--out", str(tmp_path / "out")]

    CliRunner().invoke(generate_app, args)

    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    assert [page.text for page in summary.pages] == ["Anno 1720 den 5 maj\n\nNästa stycke"]


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


def test_load_pipeline_compiles_exactly_the_transformers_repeated_blocks(tiny_flux3: Path) -> None:
    settings = TINY_SETTINGS.model_copy(update={"compile_blocks": True})

    pipe = load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=settings)

    compiled = {
        type(module).__name__
        for module in pipe.transformer.modules()
        if getattr(module, "_compiled_call_impl", None) is not None  # what nn.Module.compile sets
    }
    assert compiled == {"Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"}


def test_load_pipeline_refuses_to_compile_the_blocks_around_unfused_loras(
    tiny_flux3: Path, crop_pipe: ModularPipeline, write_nemo_lora: Callable[..., Path]
) -> None:
    linear = crop_pipe.transformer.get_submodule(LORA_MODULE)
    lora_dir = write_nemo_lora(
        LORA_MODULE, torch.zeros(2, linear.in_features), torch.zeros(linear.out_features, 2), alpha=2
    )
    settings = TINY_SETTINGS.model_copy(update={"compile_blocks": True})

    with pytest.raises(ValueError, match="only with its LoRAs fused"):
        load_pipeline(tiny_flux3, crop_blocks, loras={"v2": lora_dir}, settings=settings)


def test_draw_crop_refuses_reference_guidance_on_a_crop_without_a_reference(crop_pipe: ModularPipeline) -> None:
    guided = TINY_SETTINGS.model_copy(update={"reference_guidance_scale": 3.0})

    with pytest.raises(ValueError, match="reference guidance needs a reference image"):
        draw_crop(crop_pipe, _plan(), name="000", seed=0, reference=None, settings=guided)


def test_load_pipeline_samples_at_the_shift_asked_for_and_the_models_own_by_default(tiny_flux3: Path) -> None:
    drawn = {}
    for name, shift in (("default", None), ("own", 6.93), ("16", 16.0)):
        settings = TINY_SETTINGS.model_copy(update={"sampling_shift": shift})
        pipe = load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=settings)
        drawn[name] = draw_crop(pipe, _plan(), name="000", seed=0, reference=None, settings=settings).image.tobytes()

    assert drawn["own"] == drawn["default"]
    assert drawn["16"] != drawn["default"]


def test_draw_crop_guides_each_crop_only_until_guidance_stop(tiny_flux3: Path) -> None:
    settings = TINY_SETTINGS.model_copy(update={"guidance_stop": 0.5})
    pipe = load_pipeline(tiny_flux3, crop_blocks, loras={}, settings=settings)
    first = draw_crop(pipe, _plan(), name="000", seed=0, reference=None, settings=settings)
    passes: list[str] = []
    pipe.transformer.register_forward_hook(lambda *_: passes.append("pass"))

    again = draw_crop(pipe, _plan(), name="000", seed=0, reference=None, settings=settings)

    assert len(passes) == 3  # of the 2 steps, the first guided (2 passes) and the second not
    assert again.image.tobytes() == first.image.tobytes()


@pytest.mark.parametrize("fuse", [pytest.param(False, id="adapter"), pytest.param(True, id="fused")])
@pytest.mark.parametrize("strength", [pytest.param(1.0, id="trained"), pytest.param(0.5, id="half")])
def test_nemo_lora_loads_at_its_trained_strength_times_the_lora_scale(
    tiny_flux3: Path, crop_pipe: ModularPipeline, write_nemo_lora: Callable[..., Path], fuse: bool, strength: float
) -> None:
    torch.manual_seed(0)
    linear = crop_pipe.transformer.get_submodule(LORA_MODULE)
    a = torch.randn(2, linear.in_features)
    b = torch.randn(linear.out_features, 2)
    lora_dir = write_nemo_lora(LORA_MODULE, a, b, alpha=4)  # scale alpha / r = 2
    x = torch.randn(3, linear.in_features)
    expected = linear(x.to(linear.weight.dtype)).float() + strength * 2 * x @ a.T @ b.T
    settings = TINY_SETTINGS.model_copy(update={"fuse_loras": fuse, "lora_scale": strength})

    pipe = load_pipeline(tiny_flux3, crop_blocks, loras={"v2": lora_dir}, settings=settings)

    module = pipe.transformer.get_submodule(LORA_MODULE)
    assert (type(module) is torch.nn.Linear) == fuse  # fused: a plain Linear, the LoRA in its weight
    torch.testing.assert_close(module(x.to(linear.weight.dtype)).float(), expected, rtol=2e-2, atol=2e-1)


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
