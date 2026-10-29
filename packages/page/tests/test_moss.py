"""The page and crop pipelines with a MOSS-Image repo on CPU (conftest's tiny MOSS, random weights, so it draws noise):
a crop plan's caption compiled as MOSS's Layout prompt exactly as the verified scratchpad code compiled it, a crop drawn
end to end from its seed, the line repair in MOSS's denoise loop, MOSS's own guidance and sampling shift, what MOSS
refuses, and each repo drawn by its own family. A MOSS trained to take a reference (conftest's tiny reference training,
exported and initialised): its training cache holds what drawing encodes, it draws a crop from a reference and guides
toward it, its own training pairs drawn back (diffu_page.cli.draw_pairs) are the crops the page pipeline draws, a
described caption's look words drop to exactly the plain caption the page captions would write, and a named caption's
citation drops with the comma after its archive."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from diffusers import Flux3AutoBlocks, Flux3ModularPipeline
from diffusers.modular_pipelines import ModularPipeline, SequentialPipelineBlocks
from PIL import Image

from diffu.modular_pipelines.diffu import DiffuAutoBlocks, DiffuModularPipeline
from diffu.modular_pipelines.loading import COMPONENT_DTYPES
from diffu_moss.layout_prompt import caption_layout, caption_texts
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_moss.sample_caption import without_look, without_sample
from diffu_moss.train.cache import load_sample, read_entries
from diffu_moss.train.pairs import Pair, Variant, read_pairs
from diffu_page.captions.hand import HandLook, ReferenceLook
from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord, PageStyle
from diffu_page.captions.restyle import CaptionParts, described_edit_caption, edit_caption, named_edit_caption
from diffu_page.cli.draw_pairs import draw_pairs
from diffu_page.crops.geometry import Crop, CropSettings, crops_for_page, regions_of
from diffu_page.evalset import EvalCrop
from diffu_page.judge import LineVerdict
from diffu_page.layout import load_xml_layout
from diffu_page.pipeline import (
    FAMILIES,
    DrawSettings,
    crop_blocks,
    draw_crop,
    load_pipeline,
    model_family,
    repair_lines,
)
from diffu_page.plan import CropPlan, plan_page, plan_whole_page

if TYPE_CHECKING:
    from conftest import NemoLoraWriter, RecordMaker

# The scratchpad's moss_prompt.layout_prompt (27 Oct; the prompts MOSS's 1715 LoRA trained on) of the certificate
# page's crop: its printed and handwritten lines as text items, its paper and ruled lines as objects
LAYOUT_PROMPT = Path(__file__).parent / "data" / "moss_layout_prompt.json"
MOSS_SETTINGS = DrawSettings(steps=2, device="cpu")


class ScriptedJudge:
    """Reads every line's text right, except the lines whose text it is told to misread: those it reads as nothing."""

    def __init__(self, misread: frozenset[str] = frozenset()) -> None:
        self.misread = misread

    def __call__(self, image: Image.Image, lines: Sequence[LineRecord]) -> list[LineVerdict]:
        return [LineVerdict.of("" if line.text in self.misread else line.text, line.text) for line in lines]


@pytest.fixture(scope="module")
def moss_pipe(tiny_moss: Path) -> ModularPipeline:
    return load_pipeline(tiny_moss, crop_blocks, loras={}, settings=MOSS_SETTINGS)


@pytest.fixture
def small_plan(two_column_page: Path, page_style: PageStyle) -> CropPlan:
    """The two-column page planned whole at 64 x 32 px: six lines under a layout caption."""
    return plan_whole_page(load_xml_layout(two_column_page), page_style, pixels=64 * 32)


def _draw(pipe: ModularPipeline, plan: CropPlan, *, seed: int, reference: Image.Image | None = None) -> Image.Image:
    crop = draw_crop(
        pipe, plan, name="000", seed=seed, reference=reference, settings=MOSS_SETTINGS, judge=ScriptedJudge()
    )
    return crop.image


def _latents(pipe: ModularPipeline, plan: CropPlan, *, seed: int, **inputs: object) -> torch.Tensor:
    pipe.guider.set_state(step=0, num_inference_steps=MOSS_SETTINGS.steps, timestep=None)
    return pipe(
        crop_plan=plan,
        generator=torch.Generator().manual_seed(seed),
        **MOSS_SETTINGS.pipeline_inputs(),
        **inputs,
        output="latents",
    )


def test_moss_crop_blocks_prompt_a_crop_plan_with_its_caption_compiled_as_the_verified_code_compiled_it(
    certificate_page: Path, page_style: PageStyle
) -> None:
    plan = plan_page(load_xml_layout(certificate_page), page_style)[0]
    blocks = crop_blocks(FAMILIES["MossAutoBlocks"]).sub_blocks
    prompt_steps = SequentialPipelineBlocks.from_blocks_dict(
        {name: blocks[name] for name in ("crop_prompt", "layout_prompt")}
    ).init_pipeline()

    prompt = prompt_steps(crop_plan=plan, output="prompt")

    assert prompt == LAYOUT_PROMPT.read_text(encoding="utf-8")


def test_crop_pipeline_draws_a_plan_with_a_moss_repo_alike_from_one_seed_and_otherwise_from_another(
    moss_pipe: ModularPipeline, small_plan: CropPlan
) -> None:
    drawn = [_draw(moss_pipe, small_plan, seed=seed) for seed in (0, 0, 1)]

    assert drawn[0].size == (64, 32)
    assert drawn[1].tobytes() == drawn[0].tobytes()
    assert drawn[2].tobytes() != drawn[0].tobytes()


def test_repair_ends_the_moss_latents_outside_the_mask_on_the_drawn_crops_and_draws_inside_again(
    moss_pipe: ModularPipeline, small_plan: CropPlan
) -> None:
    drawn = _latents(moss_pipe, small_plan, seed=0)
    mask = torch.zeros(1, 1, *drawn.shape[-2:])
    mask[..., :2] = 1.0  # the crop's left half, two of its four latent columns

    redrawn = _latents(moss_pipe, small_plan, seed=1, repair_latents=drawn, repair_mask=mask)

    assert torch.equal(redrawn[..., 2:], drawn[..., 2:])
    assert not torch.equal(redrawn[..., :2], drawn[..., :2])


def test_repair_lines_draws_a_rejected_line_of_a_moss_crop_again_from_the_crops_own_latents(
    moss_pipe: ModularPipeline, small_plan: CropPlan
) -> None:
    misread = ScriptedJudge(frozenset({small_plan.lines[0].text}))
    crop = draw_crop(moss_pipe, small_plan, name="000", seed=0, reference=None, settings=MOSS_SETTINGS, judge=misread)

    repaired = repair_lines(moss_pipe, crop, seed=1, reference=None, settings=MOSS_SETTINGS, judge=ScriptedJudge())

    assert (crop.unclear_lines, repaired.unclear_lines) == (1, 0)
    assert repaired.image.tobytes() != crop.image.tobytes()


@pytest.mark.parametrize(
    ("repo", "guidance_scale"),
    [pytest.param("tiny_flux3", 4.0, id="flux3"), pytest.param("tiny_moss", 3.0, id="moss")],
)
def test_load_pipeline_guides_at_the_model_familys_own_scale_when_the_settings_name_none(
    repo: str, guidance_scale: float, request: pytest.FixtureRequest
) -> None:
    pipe = load_pipeline(request.getfixturevalue(repo), crop_blocks, loras={}, settings=MOSS_SETTINGS)

    assert pipe.guider.guidance_scale == guidance_scale


@pytest.mark.parametrize(
    ("sampling_shift", "timesteps"),
    [
        pytest.param(None, [1000.0, 24.0], id="its-own-5"),  # a second shift would step 109.5
        pytest.param(16.0, [1000.0, 204.0], id="asked-for"),  # a second shift would step 803.9
    ],
)
def test_moss_steps_its_sampling_grid_at_its_own_shift_or_the_one_asked_for(
    tiny_moss: Path, small_plan: CropPlan, sampling_shift: float | None, timesteps: list[float]
) -> None:
    settings = MOSS_SETTINGS.model_copy(update={"sampling_shift": sampling_shift})
    pipe = load_pipeline(tiny_moss, crop_blocks, loras={}, settings=settings)

    draw_crop(pipe, small_plan, name="000", seed=0, reference=None, settings=settings, judge=ScriptedJudge())

    assert pipe.scheduler.timesteps.tolist() == pytest.approx(timesteps)


def test_moss_not_trained_with_references_refuses_to_draw_a_crop_from_a_reference_image(
    moss_pipe: ModularPipeline, small_plan: CropPlan
) -> None:
    with pytest.raises(ValueError, match="this MOSS-Image model takes no reference image"):
        _draw(moss_pipe, small_plan, seed=0, reference=Image.new("RGB", (64, 32), "white"))


def _pair_plan(pair: Pair) -> CropPlan:
    """A style pair's target as diffu-evaluate draws a held-out crop of its set: its caption and lines (each line's
    text as its row quotes it, its box the row's) at the target's size."""
    texts = caption_texts(pair.caption)
    lines = [
        LineRecord(id=entry.id, text=texts[entry.id], role="", bbox=entry.bbox)
        for entry in caption_layout(pair.caption)
        if entry.id in texts
    ]
    with Image.open(pair.target) as target:
        width, height = target.size
    return EvalCrop(file_name=pair.target.name, caption=pair.caption, width=width, height=height, lines=lines).plan()


@pytest.fixture(scope="module")
def first_pair(tiny_pairs: Path) -> Pair:
    return read_pairs(tiny_pairs / "train.jsonl")[0]


def test_reference_cache_holds_what_drawing_encodes_for_the_same_crop_and_reference(
    tiny_pairs: Path, tiny_pair_cache: Path, tiny_reference_moss: Path
) -> None:
    entry = next(entry for entry in read_entries(tiny_pair_cache) if entry.variant is Variant.LOOK)
    pair = next(pair for pair in read_pairs(tiny_pairs / "train.jsonl") if pair.id == entry.pair)
    blocks = crop_blocks(FAMILIES["MossAutoBlocks"]).sub_blocks
    names = ("crop_prompt", "reference_grid", "layout_prompt", "text_encoder", "reference_encoder")
    steps = SequentialPipelineBlocks.from_blocks_dict({name: blocks[name] for name in names})
    pipe = MossModularPipeline(blocks=steps, pretrained_model_name_or_path=str(tiny_reference_moss))
    pipe.load_components(names=["text_encoder", "tokenizer", "vae"], dtype=COMPONENT_DTYPES)

    drawn = pipe(
        crop_plan=_pair_plan(pair), image=Image.open(pair.reference), output=["prompt_embeds", "reference_latents"]
    )

    cached = load_sample(tiny_pair_cache, entry, "cpu")
    assert torch.equal(cached.prompt_embeds, drawn["prompt_embeds"])
    assert cached.reference is not None
    assert torch.equal(cached.reference, drawn["reference_latents"][0].to(cached.reference.dtype))


def test_reference_trained_moss_draws_a_crop_from_its_reference_through_the_page_pipeline(
    tiny_reference_moss: Path, first_pair: Pair
) -> None:
    pipe = load_pipeline(tiny_reference_moss, crop_blocks, loras={}, settings=MOSS_SETTINGS)
    plan = _pair_plan(first_pair)
    # another sample under the same prompt: a reference also changes the prompt, which names the sample image
    from_blank = _draw(pipe, plan, seed=0, reference=Image.new("RGB", (32, 32), "white"))

    referenced = _draw(pipe, plan, seed=0, reference=Image.open(first_pair.reference))

    assert referenced.size == (64, 32)
    assert referenced.tobytes() != from_blank.tobytes()


def test_draw_pairs_draws_a_trained_pair_back_as_the_page_pipeline_draws_its_crop_from_its_reference(
    tiny_pairs: Path, tiny_pair_cache: Path, tiny_reference_moss: Path, tmp_path: Path
) -> None:
    sheets = draw_pairs(tiny_reference_moss, tiny_pair_cache, tmp_path, count=3, seed=0, settings=MOSS_SETTINGS)

    # the cache's first three pairs drawn with their reference: vol_01, cached without it, is passed over
    assert [sheet.name for sheet in sheets] == ["000_vol_08.jpg", "001_vol_09.jpg", "002_vol_06.jpg"]
    pair = next(pair for pair in read_pairs(tiny_pairs / "train.jsonl") if pair.id == "vol_08")  # cached in look form
    pipe = load_pipeline(tiny_reference_moss, crop_blocks, loras={}, settings=MOSS_SETTINGS)
    crop = _draw(pipe, _pair_plan(pair), seed=0, reference=Image.open(pair.reference))
    with Image.open(tmp_path / "generated" / "000.png") as drawn:
        assert drawn.tobytes() == crop.tobytes()


def test_reference_trained_moss_guides_toward_its_reference_with_a_pass_without_it(
    tiny_reference_moss: Path, first_pair: Pair
) -> None:
    guided_settings = MOSS_SETTINGS.model_copy(update={"reference_guidance_scale": 3.0})
    plain, guided = (
        load_pipeline(tiny_reference_moss, crop_blocks, loras={}, settings=settings)
        for settings in (MOSS_SETTINGS, guided_settings)
    )
    plan, reference = _pair_plan(first_pair), Image.open(first_pair.reference)
    unguided = _draw(plain, plan, seed=0, reference=reference)

    drawn = draw_crop(
        guided, plan, name="000", seed=0, reference=reference, settings=guided_settings, judge=ScriptedJudge()
    ).image

    assert drawn.tobytes() != unguided.tobytes()


@pytest.fixture
def certificate_crop(certificate_page: Path) -> Crop:
    """The crop of a printed certificate filled in by hand: printed lines, handwritten ones and ruled lines."""
    layout = load_xml_layout(certificate_page)
    rules = [(rule.x0, rule.y0, rule.x1, rule.y1) for rule in layout.rules]
    return crops_for_page(regions_of(layout), (layout.width, layout.height), CropSettings(), rules=rules)[0]


@pytest.mark.parametrize(
    ("crop", "reference"),
    [
        pytest.param(
            "body_and_notes_crop",
            ReferenceLook(
                scan=ScanLook(greyscale=False, paper="cream", ink="dark brown"),
                hand=HandLook(slant=33.0, stroke=0.05, letters=0.3, density=0.2, paper="#D8C8A8", ink="#4A3828"),
            ),
            id="colour-scan-margin-notes",
        ),
        pytest.param(
            "certificate_crop",
            ReferenceLook(
                scan=ScanLook(greyscale=True, paper="light grey", ink="black"),
                hand=HandLook(slant=None, stroke=0.09, letters=0.4, density=0.15, paper="#C8C8C8", ink="#202020"),
            ),
            id="greyscale-scan-unmeasured-slant-printed-lines-and-rules",
        ),
    ],
)
def test_without_look_turns_a_described_reference_caption_into_the_plain_caption_of_the_same_crop(
    crop: str, reference: ReferenceLook, make_record: RecordMaker, request: pytest.FixtureRequest
) -> None:
    parts = CaptionParts.of(make_record(request.getfixturevalue(crop)))

    plain = without_look(described_edit_caption(parts, reference))

    assert plain == edit_caption(parts)


def test_without_sample_drops_a_named_reference_captions_citation_with_the_comma_after_its_archive(
    body_and_notes_crop: Crop, make_record: RecordMaker
) -> None:
    named = named_edit_caption(CaptionParts.of(make_record(body_and_notes_crop)))

    caption = without_sample(named)

    assert caption.startswith(
        "Create a flat, evenly lit archival scan of a new handwritten page from the court minutes of Arjeplogs "
        "tingslag, dated 1690–1700. The yellowed paper <paper_1> carries"
    )


def test_load_pipeline_refuses_a_lora_for_a_moss_repo(tiny_moss: Path, write_nemo_lora: NemoLoraWriter) -> None:
    lora = write_nemo_lora("layers.0.proj.q", torch.zeros(2, 32), torch.zeros(32, 2), alpha=2)

    with pytest.raises(ValueError, match="takes no LoRAs at draw time: merge them into the transformer"):
        load_pipeline(tiny_moss, crop_blocks, loras={"hand": lora}, settings=MOSS_SETTINGS)


@pytest.mark.parametrize(
    ("repo", "blocks", "pipeline"),
    [
        pytest.param("tiny_flux3", Flux3AutoBlocks, Flux3ModularPipeline, id="flux3"),
        pytest.param("tiny_diffu", DiffuAutoBlocks, DiffuModularPipeline, id="diffu"),
        pytest.param("tiny_moss", MossAutoBlocks, MossModularPipeline, id="moss"),
    ],
)
def test_model_family_draws_each_repo_with_the_blocks_and_pipeline_of_its_own_family(
    repo: str, blocks: type, pipeline: type, request: pytest.FixtureRequest
) -> None:
    family = model_family(request.getfixturevalue(repo))

    assert (family.blocks, family.pipeline) == (blocks, pipeline)
