"""The page and crop pipelines with a MOSS-Image repo on CPU (conftest's tiny MOSS, random weights, so it draws noise):
a crop plan's caption compiled as MOSS's Layout prompt exactly as the verified scratchpad code compiled it, a crop drawn
end to end from its seed, the line repair in MOSS's denoise loop, MOSS's own guidance and sampling shift, what MOSS
refuses, and each repo drawn by its own family."""

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
from diffu_moss.modular_pipelines.moss import MossAutoBlocks, MossModularPipeline
from diffu_page.captions.prompt import LineRecord, PageStyle
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
    from conftest import NemoLoraWriter

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


def test_moss_refuses_to_draw_a_crop_from_a_reference_image(moss_pipe: ModularPipeline, small_plan: CropPlan) -> None:
    with pytest.raises(ValueError, match="MOSS-Image draws from its prompt alone: it takes no reference image"):
        _draw(moss_pipe, small_plan, seed=0, reference=Image.new("RGB", (64, 32), "white"))


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
