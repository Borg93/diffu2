"""The page and crop pipelines end to end on CPU: diffusers' tiny FLUX.3 test model, the real HTR read-back, no GPU.

The tiny model draws noise, so every line reads back badly: what is checked is the wiring (a plan in, an image of the
plan's size with one label per line out; a reference image drawn from as it was resized for training, and guided
toward when asked; a glyph model's crop drawn from its plan's glyph map; diffu-generate drawing each planned crop
on its own, so one that fails costs no other, and each hand's pages with its own pipeline) and that a NeMo LoRA
loads at its trained strength.
"""

from __future__ import annotations

import json
import weakref
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

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
from diffu_page.cli.generate import PageGroup, Summary, draw_pages, plan_jobs
from diffu_page.cli.generate import app as generate_app
from diffu_page.crops.grid import as_reference
from diffu_page.glyphs import glyph_map
from diffu_page.jobs import PageJob, crop_jobs
from diffu_page.judge import LineVerdict
from diffu_page.labels import PageElements, to_grounded_markdown, to_markdown
from diffu_page.layout import PageSide
from diffu_page.local_run import NOT_RUN
from diffu_page.lora import ADAPTER_CONFIG, ADAPTER_WEIGHTS, nemo_lora_state_dict
from diffu_page.marking import PrintHandClassifier
from diffu_page.output import jpeg_bytes
from diffu_page.pipeline import (
    FAMILIES,
    BlocksFor,
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

if TYPE_CHECKING:
    from conftest import NemoLoraWriter, ScannedFormMaker

TINY_SETTINGS = DrawSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
TINY_OPTIONS = ["--steps", "2", "--text-layer", "1", "--device", "cpu"]  # TINY_SETTINGS as diffu-generate's options
FLAG_PAGES = ["--templates", "page.xml", "--text-from", "text.txt"]  # a page from the flags, in a test's working dir
LORA_MODULE = "single_transformer_blocks.0.attn.to_q"
FLAG_STYLE = PageStyle(  # diffu-generate --title "the archive" with every other style flag left at its default
    provenance=Provenance(title="the archive", years=""),
    look=ScanLook(greyscale=False, paper=Paper.CREAM, ink=Ink.DARK_BROWN),
)
PRESET_STYLE = PageStyle(  # the style of every test preset: the 1715 letter book's, as its hand evaluations name it
    provenance=Provenance(title="the records of Överståthållarämbetet", years=""),
    look=ScanLook(greyscale=False, paper=Paper.YELLOWED, ink=Ink.BROWN),
)


def _write_presets(path: Path, **presets: dict[str, object]) -> Path:
    """A presets file at ``path`` of each preset given, in PRESET_STYLE with the fields given (its lora, reference)."""
    style = PRESET_STYLE.model_dump(mode="json")
    path.write_text(json.dumps({name: {**style, **fields} for name, fields in presets.items()}), encoding="utf-8")
    return path


def _zero_lora(write_nemo_lora: NemoLoraWriter) -> Path:
    """A NeMo LoRA that changes nothing: for a run that plans no crop, and so loads none."""
    return write_nemo_lora(LORA_MODULE, torch.zeros(2, 8), torch.zeros(8, 2), alpha=2)


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


def _page_geometry(labels: Path) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """A PAGE label file's page size, and the x-extent of each line's outline in reading order."""
    root = ET.parse(labels).getroot()
    page = next(e for e in root.iter() if e.tag.endswith("}Page"))
    outlines = [
        next(e for e in line if e.tag.endswith("}Coords")).get("points", "")
        for line in root.iter()
        if line.tag.endswith("}TextLine")
    ]
    xs = [[int(point.split(",")[0]) for point in outline.split()] for outline in outlines]
    return (int(page.get("imageWidth", "")), int(page.get("imageHeight", ""))), [(min(x), max(x)) for x in xs]


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


@pytest.mark.parametrize(
    ("family", "glyph_steps", "loop"),
    [
        pytest.param("Flux3AutoBlocks", [], ["denoiser", "after_denoiser"], id="flux3"),
        pytest.param(
            "DiffuAutoBlocks",
            ["prepare_glyph"],
            ["glyph_concat", "denoiser", "glyph_split", "after_denoiser"],
            id="diffu",
        ),
    ],
)
def test_crop_blocks_leave_the_model_familys_own_blocks_as_they_are(
    family: str, glyph_steps: list[str], loop: list[str]
) -> None:
    crop_blocks(FAMILIES[family])

    own = list(FAMILIES[family].blocks().sub_blocks["denoise"].sub_blocks.values())
    assert [list(branch.sub_blocks) for branch in own] == [
        ["input", "cond_input", "prepare_latents", *glyph_steps, "prepare_image_cond_latents", "set_timesteps"]
        + ["prepare_rope_inputs", "denoise", "after_denoise"],
        ["input", "prepare_latents", *glyph_steps, "set_timesteps", "prepare_rope_inputs", "denoise", "after_denoise"],
    ]
    assert [list(branch.sub_blocks["denoise"].sub_blocks) for branch in own] == [loop, loop]


def _model_blocks_only(family: ModelFamily) -> CropBlocks:
    """Our crop steps before the model's own blocks, as they stand: no glyph map drawn, no line repair."""
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


def _latents(pipe: ModularPipeline, seed: int, plan: CropPlan | None = None, **inputs: object) -> torch.Tensor:
    pipe.guider.set_state(step=0, num_inference_steps=TINY_SETTINGS.steps, timestep=None)
    return pipe(
        crop_plan=plan or _plan(),
        generator=torch.Generator().manual_seed(seed),
        **TINY_SETTINGS.pipeline_inputs(),
        **inputs,
        output="latents",
    )


def test_crop_pipeline_draws_a_glyph_models_crop_from_the_glyph_map_of_its_plan(tiny_diffu_glyph: Path) -> None:
    plan = _plan().model_copy(update={"rect": (0, 0, 128, 64), "width": 128, "height": 64})  # sides of 64 px or more
    own = load_pipeline(tiny_diffu_glyph, _model_blocks_only, loras={}, settings=TINY_SETTINGS)
    from_its_map = _latents(own, seed=0, plan=plan, glyph_image=glyph_map(plan.lines, (128, 64)))
    pipe = load_pipeline(tiny_diffu_glyph, crop_blocks, loras={}, settings=TINY_SETTINGS)

    drawn = _latents(pipe, seed=0, plan=plan)

    assert torch.equal(drawn, from_its_map)


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
    crop_pipe: ModularPipeline,
    two_column_page: Path,
    wide_and_narrow_body_page: Path,
    tmp_path: Path,
    page_style: PageStyle,
    page_text: str,
) -> None:
    unreadable = Reference(image=tmp_path / "missing.png")  # planned without it; each of its crops fails to read it
    pages = [
        PageJob(template=two_column_page, text=page_text, style=page_style, reference=unreadable),
        PageJob(template=wide_and_narrow_body_page, text=page_text, style=page_style),
        PageJob(template=wide_and_narrow_body_page, text=page_text, style=page_style),
    ]
    unloadable = {"hand": tmp_path / "hand"}  # the last page's group, drawn first, whose pipeline fails to load

    def load(loras: dict[str, Path]) -> ModularPipeline:
        if loras == unloadable:
            raise ValueError(f"{loras['hand']} is not a NeMo LoRA model dir")
        return crop_pipe

    jobs, unplanned = plan_jobs(enumerate(pages))
    *drawable, last = jobs

    crops, failed = draw_pages(
        load, [(unloadable, last), *(({}, job) for job in drawable)], settings=TINY_SETTINGS, out=tmp_path, prof=None
    )

    assert unplanned == {}
    assert [crop.name for crop in crops] == ["001_wide_and_narrow_0"]
    assert {name: error.split(":")[0] for name, error in failed.items()} == {
        "000_two_columns_0": "FileNotFoundError",
        "000_two_columns_1": "FileNotFoundError",
        "002_wide_and_narrow_0": "ValueError",
    }
    assert sorted(path.name for path in tmp_path.glob("0*")) == [
        "001_wide_and_narrow_0.jpg",
        "001_wide_and_narrow_0.xml",
    ]


def test_draw_pages_stops_when_the_device_fails_and_loads_no_later_group(
    crop_pipe: ModularPipeline,
    two_column_page: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page_style: PageStyle,
    page_text: str,
) -> None:
    jobs, _ = plan_jobs(enumerate([PageJob(template=two_column_page, text=page_text, style=page_style)]))
    loaded: list[dict[str, Path]] = []

    def load(loras: dict[str, Path]) -> ModularPipeline:
        loaded.append(loras)
        return crop_pipe

    def out_of_memory(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(crop_pipe.transformer, "forward", out_of_memory)
    later = {"hand": tmp_path / "hand"}  # the second crop's group

    crops, failed = draw_pages(load, [({}, jobs[0]), (later, jobs[1])], settings=TINY_SETTINGS, out=tmp_path, prof=None)

    assert (crops, failed, loaded) == (
        [],
        {
            "000_two_columns_0": "OutOfMemoryError: CUDA out of memory",
            "000_two_columns_1": NOT_RUN,
        },
        [{}],
    )


class FirstReadMisses:
    """Misreads every line the first time it reads, then reads every line right: a drawing's lines are all rejected,
    their redraw's all passed."""

    def __init__(self) -> None:
        self.reads = 0

    def __call__(self, image: Image.Image, lines: Sequence[LineRecord]) -> list[LineVerdict]:
        self.reads += 1
        return [LineVerdict.of("" if self.reads == 1 else line.text, line.text) for line in lines]


def test_draw_pages_draws_the_rejected_lines_again_and_counts_them_repaired(
    crop_pipe: ModularPipeline,
    wide_and_narrow_body_page: Path,
    tmp_path: Path,
    page_style: PageStyle,
    page_text: str,
) -> None:
    (job,), _ = plan_jobs(enumerate([PageJob(template=wide_and_narrow_body_page, text=page_text, style=page_style)]))
    settings = TINY_SETTINGS.model_copy(update={"repair_passes": 1})

    crops, failed = draw_pages(
        lambda loras: crop_pipe, [({}, job)], settings=settings, out=tmp_path, prof=None, judge=FirstReadMisses()
    )

    lines = len(job.plan.lines)
    assert failed == {}
    assert [(crop.unclear_lines, crop.repaired) for crop in crops] == [(0, lines)]


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


def test_generate_records_each_page_it_cannot_plan_without_loading_the_model(
    one_line_page: Path, body_and_notes_page: Path, tmp_path: Path
) -> None:
    no_model = tmp_path / "no_model"  # never read: no page plans to a crop
    missing = tmp_path / "missing.xml"  # a template that cannot be read, so not split into its pages either
    args = ["--templates", str(one_line_page), "--templates", str(missing), "--text-from", str(body_and_notes_page)]
    args += ["--title", "the archive", "--model", str(no_model), "--out", str(tmp_path)]
    args += ["--steps", "2", "--text-layer", "1", "--device", "cpu"]

    result = CliRunner().invoke(generate_app, args)

    summary = Summary.model_validate_json((tmp_path / "summary.json").read_text())
    assert result.exit_code == 1
    assert (summary.model, summary.settings, summary.crops) == (
        str(no_model),
        DrawSettings(steps=2, text_layers=(1,), device="cpu"),
        [],
    )
    assert [(page.template, page.style, page.seed) for page in summary.pages] == [
        (one_line_page, FLAG_STYLE, 0),  # the style the flags name, with their defaults
        (missing, FLAG_STYLE, 0),
    ]
    assert summary.failed == {
        "000_one_line": f"ValueError: {one_line_page} plans to no crops: no window of it holds enough whole lines",
        "001_missing": f"FileNotFoundError: [Errno 2] No such file or directory: '{missing}'",
    }


def test_generate_takes_each_page_of_a_job_file_with_its_own_text_name_and_style_spelled_out_or_named(
    one_line_page: Path, tmp_path: Path, page_style: PageStyle, page_text: str, write_nemo_lora: NemoLoraWriter
) -> None:
    run_lora, hand = _zero_lora(write_nemo_lora), _zero_lora(write_nemo_lora)
    styles = _write_presets(tmp_path / "styles.json", a={"lora": hand.name, "reference": {"image": "ref.jpg"}})
    pages = [
        PageJob(template=one_line_page, text=page_text, style=page_style, id="vol7_p12"),
        PageJob(template=one_line_page, text="den 4 maj", style=FLAG_STYLE, seed=3),
    ]
    named = [
        {"template": str(one_line_page), "text": "den 5 maj", "style": "a"},
        {"template": str(one_line_page), "text": "den 6 maj", "style": "a", "reference": {"image": "/scans/own.jpg"}},
    ]
    job_file = tmp_path / "run.jsonl"
    job_file.write_text(
        "".join(page.model_dump_json() + "\n" for page in pages) + "".join(json.dumps(line) + "\n" for line in named),
        encoding="utf-8",
    )
    args = ["--jobs", str(job_file), "--styles", str(styles), "--lora", f"v2={run_lora}"]
    args += ["--model", str(tmp_path / "no_model"), "--out", str(tmp_path / "out")]

    result = CliRunner().invoke(generate_app, args)

    assert result.exit_code == 1  # no page plans to a crop, so no model loads
    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    presets_reference, own = Reference(image=tmp_path / "ref.jpg"), Reference(image=Path("/scans/own.jpg"))
    assert summary.pages == [
        *pages,
        PageJob(template=one_line_page, text="den 5 maj", style=PRESET_STYLE, reference=presets_reference),
        PageJob(template=one_line_page, text="den 6 maj", style=PRESET_STYLE, reference=own),
    ]
    assert summary.groups == [
        PageGroup(loras={"v2": run_lora}, pages=[0, 1]),
        PageGroup(loras={"hand": hand}, pages=[2, 3]),
    ]
    no_crops = f"ValueError: {one_line_page} plans to no crops: no window of it holds enough whole lines"
    assert summary.failed == dict.fromkeys(["vol7_p12", "001_one_line", "002_one_line", "003_one_line"], no_crops)


def test_generate_draws_the_templates_in_the_style_preset_style_names_by_its_hand(
    one_line_page: Path, tmp_path: Path, write_nemo_lora: NemoLoraWriter
) -> None:
    hand = _zero_lora(write_nemo_lora)
    styles = _write_presets(tmp_path / "styles.json", a={"lora": str(hand), "reference": {"image": "ref.jpg"}})
    text_file = tmp_path / "text.txt"
    text_file.write_text("den 4 maj", encoding="utf-8")
    args = ["--templates", str(one_line_page), "--text-from", str(text_file), "--styles", str(styles), "--style", "a"]
    args += ["--reference", str(tmp_path / "own.jpg"), "--model", str(tmp_path / "no_model"), "--out", str(tmp_path)]

    result = CliRunner().invoke(generate_app, args)

    assert result.exit_code == 1  # the page plans to no crop, so no model loads
    summary = Summary.model_validate_json((tmp_path / "summary.json").read_text())
    own = Reference(image=tmp_path / "own.jpg")  # --reference, in place of the preset's
    assert summary.pages == [PageJob(template=one_line_page, text="den 4 maj", style=PRESET_STYLE, reference=own)]
    assert summary.groups == [PageGroup(loras={"hand": hand}, pages=[0])]


def test_generate_draws_each_hands_pages_with_its_own_pipeline_and_keeps_their_names_and_seeds(
    tiny_flux3: Path,
    crop_pipe: ModularPipeline,
    one_line_page: Path,
    tmp_path: Path,
    write_nemo_lora: NemoLoraWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    linear = crop_pipe.transformer.get_submodule(LORA_MODULE)
    torch.manual_seed(0)
    hands = [
        write_nemo_lora(LORA_MODULE, torch.randn(2, linear.in_features), torch.randn(linear.out_features, 2), alpha=4)
        for _ in range(2)
    ]
    styles = _write_presets(tmp_path / "styles.json", a={"lora": str(hands[0])}, b={"lora": str(hands[1])})
    page = PageJob(template=one_line_page, text="den 4 maj", style=PRESET_STYLE, page_pixels=256 * 256)
    named = [{**page.model_dump(mode="json"), "style": name} for name in ("a", "b", "a")]  # pages 0 and 2 in hand a
    job_file = tmp_path / "run.jsonl"
    job_file.write_text("".join(json.dumps(line) + "\n" for line in named), encoding="utf-8")
    loads: list[tuple[dict[str, Path], int]] = []  # each load's LoRAs, and how many pipelines before it still live
    pipes: list[weakref.ref[ModularPipeline]] = []

    def load_and_count(
        model: Path, blocks: BlocksFor, *, loras: dict[str, Path], settings: DrawSettings
    ) -> ModularPipeline:
        loads.append((loras, sum(pipe() is not None for pipe in pipes)))
        pipe = load_pipeline(model, blocks, loras=loras, settings=settings)
        pipes.append(weakref.ref(pipe))
        return pipe

    monkeypatch.setattr("diffu_page.cli.generate.load_pipeline", load_and_count)
    monkeypatch.setattr("diffu_page.cli.generate.TrOCRJudge", lambda device: ScriptedJudge())  # reads are not tested
    args = ["--jobs", str(job_file), "--styles", str(styles), "--model", str(tiny_flux3)]
    args += ["--out", str(tmp_path / "out"), *TINY_OPTIONS]

    result = CliRunner().invoke(generate_app, args)

    assert result.exit_code == 0
    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    assert loads == [({"hand": hands[0]}, 0), ({"hand": hands[1]}, 0)]  # a load per hand, the one before released
    assert summary.groups == [
        PageGroup(loras={"hand": hands[0]}, pages=[0, 2]),
        PageGroup(loras={"hand": hands[1]}, pages=[1]),
    ]
    jobs = [(hands[0], crop_jobs(page, 0)[0]), (hands[0], crop_jobs(page, 2)[0]), (hands[1], crop_jobs(page, 1)[0])]
    assert [crop.name for crop in summary.crops] == [job.name for _, job in jobs]  # named as pages 0, 2 and 1
    by_hand = {
        hand: load_pipeline(tiny_flux3, crop_blocks, loras={"hand": hand}, settings=TINY_SETTINGS) for hand in hands
    }
    expected = [  # each page drawn by its hand, from its seed as that page of the run
        draw_crop(
            by_hand[hand],
            job.plan,
            name=job.name,
            seed=job.seed,
            reference=None,
            settings=TINY_SETTINGS,
            judge=ScriptedJudge(),
        )
        for hand, job in jobs
    ]
    drawn = [(tmp_path / "out" / f"{job.name}.jpg").read_bytes() for _, job in jobs]
    assert drawn == [jpeg_bytes(crop.image) for crop in expected]


def test_generate_draws_an_opening_whole_as_its_two_pages_each_labelled_in_its_own_coordinates(
    tiny_flux3: Path, opening_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_text: str
) -> None:
    text_file = tmp_path / "text.txt"
    text_file.write_text(page_text, encoding="utf-8")
    monkeypatch.setattr("diffu_page.cli.generate.TrOCRJudge", lambda device: ScriptedJudge())  # reads are not tested
    args = ["--templates", str(opening_page), "--text-from", str(text_file), "--title", "the archive", "--whole-page"]
    args += [
        "--page-megapixels",
        "0.25",
        "--format",
        "page",
        "--model",
        str(tiny_flux3),
        "--out",
        str(tmp_path / "out"),
    ]

    result = CliRunner().invoke(generate_app, [*args, *TINY_OPTIONS])

    assert result.exit_code == 0
    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    assert summary.pages == [
        PageJob(template=opening_page, text=page_text.strip(), style=FLAG_STYLE, page_pixels=250_000, side=side)
        for side in (PageSide.LEFT, PageSide.RIGHT)
    ]
    # each page drawn at about 0.25 MP (the left one, 1275 x 1200 px, at 512 x 480; the right, 1225 x 1200, at
    # 512 x 496) and each line labelled where it stands on its page: its slot, its x-extent and 15 px (0.15 of the
    # 100 px pitch) either side, so the right page's lines, x 1400-2400 of the scan and 125-1125 of the page, at 46-477
    assert [(crop.name, *_page_geometry(tmp_path / "out" / f"{crop.name}.xml")) for crop in summary.crops] == [
        ("000_opening_left_0", (512, 480), [(34, 408)] * 4 + [(416, 468)]),  # four lines, then the inner margin's note
        ("001_opening_right_0", (512, 496), [(46, 477)] * 3),
    ]


def _page_lines(labels: Path) -> list[tuple[str, tuple[int, int, int, int]]]:
    """A PAGE label file's lines in reading order: each one's text and the box of its outline."""
    lines: list[tuple[str, tuple[int, int, int, int]]] = []
    for line in (e for e in ET.parse(labels).iter() if e.tag.endswith("}TextLine")):
        outline = next(e for e in line if e.tag.endswith("}Coords")).get("points", "")
        points = [(int(x), int(y)) for x, y in (point.split(",") for point in outline.split())]
        xs, ys = [x for x, _ in points], [y for _, y in points]
        text = next(e for e in line.iter() if e.tag.endswith("}Unicode")).text or ""
        lines.append((text, (min(xs), min(ys), max(xs), max(ys))))
    return lines


def test_generate_writes_each_crops_reader_labels_beside_it_from_the_labels_its_xml_holds(
    tiny_flux3: Path, certificate_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_text: str
) -> None:
    text_file = tmp_path / "text.txt"
    text_file.write_text(page_text, encoding="utf-8")
    monkeypatch.setattr("diffu_page.cli.generate.TrOCRJudge", lambda device: ScriptedJudge())  # reads are not tested
    args = ["--templates", str(certificate_page), "--text-from", str(text_file), "--title", "the archive"]
    args += ["--whole-page", "--page-megapixels", "0.25", "--format", "page", "--reader-labels"]
    args += ["--model", str(tiny_flux3), "--out", str(tmp_path / "out")]

    result = CliRunner().invoke(generate_app, [*args, *TINY_OPTIONS])

    crop = "000_certificate_0"  # a printed heading, and each printed label and its field a region of their own
    out = tmp_path / "out"
    assert result.exit_code == 0
    assert sorted(path.name for path in out.glob(f"{crop}.*")) == [
        f"{crop}{suffix}" for suffix in (".elements.json", ".grounded.md", ".jpg", ".md", ".xml")
    ]
    page = PageElements.model_validate_json((out / f"{crop}.elements.json").read_text(encoding="utf-8"))
    assert [(line.text, line.box) for element in page.elements for line in element.lines] == _page_lines(
        out / f"{crop}.xml"
    )
    assert [(out / f"{crop}{suffix}").read_text(encoding="utf-8") for suffix in (".md", ".grounded.md")] == [
        f"{to_markdown(page)}\n",
        f"{to_grounded_markdown(page)}\n",
    ]


def _productions_and_rules(labels: Path) -> tuple[list[tuple[str, str]], int]:
    """A PAGE label file's lines that are not running text (their text and production), and its ruled lines' count."""
    root = ET.parse(labels).getroot()
    lines = [
        (
            next(e for e in line.iter() if e.tag.endswith("}Unicode")).text or "",
            line.get("production", ""),
        )
        for region in root.iter()
        if region.tag.endswith("}TextRegion") and region.get("type") == "heading"
        for line in region
        if line.tag.endswith("}TextLine")
    ]
    return lines, sum(1 for e in root.iter() if e.tag.endswith("}SeparatorRegion"))


def test_generate_marks_each_template_from_the_scan_beside_it_when_asked(
    tiny_flux3: Path,
    make_scanned_form: ScannedFormMaker,
    ink_meter: PrintHandClassifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    page_text: str,
) -> None:
    form = make_scanned_form()
    alone = tmp_path / "alone.xml"  # the same form, with no scan beside it
    alone.write_text(form.read_text(encoding="utf-8"), encoding="utf-8")
    weights = tmp_path / "print_hand.pt"
    weights.touch()
    monkeypatch.setattr(
        PrintHandClassifier, "load", classmethod(lambda cls, path: ink_meter if path == weights else None)
    )
    monkeypatch.setattr("diffu_page.cli.generate.TrOCRJudge", lambda device: ScriptedJudge())  # reads are not tested
    text_file = tmp_path / "text.txt"
    text_file.write_text(page_text, encoding="utf-8")
    args = [
        "--templates",
        str(form),
        "--templates",
        str(alone),
        "--text-from",
        str(text_file),
        "--title",
        "the archive",
    ]
    args += ["--whole-page", "--page-megapixels", "0.25", "--format", "page", "--mark-from-scan"]
    args += ["--print-hand-model", str(weights), "--model", str(tiny_flux3), "--out", str(tmp_path / "out")]

    result = CliRunner().invoke(generate_app, [*args, *TINY_OPTIONS])

    assert result.exit_code == 1  # the form with no scan beside it cannot be marked; the run goes on
    summary = Summary.model_validate_json((tmp_path / "out" / "summary.json").read_text())
    assert summary.marked_with == weights
    assert summary.failed == {
        "001_alone": f"FileNotFoundError: {alone} has no scan beside it to be marked from (its imageFilename, one "
        "folder up)"
    }
    assert [crop.name for crop in summary.crops] == ["000_form_0"]
    assert _productions_and_rules(tmp_path / "out" / "000_form_0.xml") == ([("Mantalslängd 1765", "printed")], 4)


def test_generate_refuses_to_mark_templates_without_a_print_hand_classifier_before_it_loads(
    one_line_page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # short paths: rich wraps the error panel at 80 columns, inside a long one
    args = ["--templates", str(one_line_page), "--text-from", str(one_line_page), "--title", "the archive"]
    args += ["--mark-from-scan", "--print-hand-model", "none.pt", "--model", "no_model", "--out", "out"]

    result = CliRunner().invoke(generate_app, args)

    message = " ".join(result.output.replace("│", " ").split())
    assert result.exit_code == 2  # a usage error, raised before the (missing) model would have been read
    assert "no print/hand classifier at none.pt" in message


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        pytest.param(["--jobs", "run.jsonl"], "run.jsonl:1: no style preset is named 'b' (the presets: a)", id="job"),
        pytest.param([*FLAG_PAGES, "--style", "b"], "no style preset is named 'b' (the presets: a)", id="flags"),
        pytest.param(
            [*FLAG_PAGES, "--style", "a", "--ink", "black"], "--ink: the style preset gives its own", id="look"
        ),
        pytest.param(
            [*FLAG_PAGES, "--style", "a", "--lora", "v2=lora_0"],
            "every page names a style preset, which gives the LoRAs it is drawn with",
            id="lora",
        ),
    ],
)
def test_generate_refuses_a_named_style_it_has_no_preset_of_or_given_twice_before_it_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_nemo_lora: NemoLoraWriter, options: list[str], refusal: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_presets(tmp_path / "styles.json", a={"lora": _zero_lora(write_nemo_lora).name})  # lora_0
    job = {"template": "page.xml", "text": "den 4 maj", "style": "b"}
    (tmp_path / "run.jsonl").write_text(json.dumps(job) + "\n", encoding="utf-8")
    (tmp_path / "text.txt").write_text("den 4 maj", encoding="utf-8")
    args = ["--styles", "styles.json", "--model", str(tmp_path / "no_model"), "--out", "out", *options]

    result = CliRunner().invoke(generate_app, args)

    message = " ".join(result.output.replace("│", " ").split())  # rich wraps the error panel at 80 columns
    assert result.exit_code == 2  # a usage error, raised before the (missing) model would have been read
    assert refusal in message


@pytest.mark.parametrize(
    ("options", "refusal"),
    [
        pytest.param(
            ["--jobs", "run.jsonl", "--title", "an archive", "--style", "a"],
            "--title, --style: the job file gives each page its own",
            id="jobs-and-flags",
        ),
        pytest.param([], "give --jobs, or --templates, --text-from and --title or --style", id="no-pages"),
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
    tiny_flux3: Path, crop_pipe: ModularPipeline, write_nemo_lora: NemoLoraWriter
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
    tiny_flux3: Path, crop_pipe: ModularPipeline, write_nemo_lora: NemoLoraWriter, fuse: bool, strength: float
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
    write_nemo_lora: NemoLoraWriter, spoil: Callable[[Path], None], error: str
) -> None:
    lora_dir = write_nemo_lora(LORA_MODULE, torch.randn(2, 8), torch.randn(8, 2), alpha=4)
    spoil(lora_dir)

    with pytest.raises(ValueError, match=error):
        nemo_lora_state_dict(lora_dir)
