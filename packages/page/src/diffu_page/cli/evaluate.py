"""Evaluate a model (FLUX.3 + LoRAs, or diffu) on held-out crops: draw each from its stored caption with the crop
pipeline (:func:`diffu_page.pipeline.crop_blocks`) and read every line back, beside the real crop's read-back.

A crop that names a ``reference`` image is drawn conditioned on it (v3, style by example). A crop's optional ``seed``
overrides ``--seed + index`` (the steering test draws every variant of a template from the same noise). Every drawn
hand is measured beside the real crop's (``diffu_page.captions.hand``) and, when the crop lists its reference's line
boxes (``reference_lines``), beside the reference's: the summary's medians show whether the drawings take their hand
from the reference at all (``HandFidelity``). Writes contact sheets (``[reference,] real, generated``),
``generated/NNN.png`` and ``results.json``; a crop that fails is recorded there and the rest still run.

    diffu-evaluate --crops data_flux3/layout_crops_v2/val_eval --lora v2=<ckpt>/model --out <dir> -n 24
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, linear_regression, median, pstdev
from typing import Annotated

import typer
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image, ImageDraw
from pydantic import BaseModel

from diffu_page.captions.hand import Box, HandDistance, HandLook, hand_distance, hand_look, lab
from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.options import (
    DEFAULTS,
    Device,
    Guidance,
    Lora,
    MaxLineCer,
    ReferenceGuidance,
    Steps,
    TextLayers,
    parse_lora,
    required_model,
)
from diffu_page.crops.grid import as_reference
from diffu_page.output import console, err_console, load_rgb
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.plan import CropPlan
from diffu_page.readback import pixel_box, read_lines
from diffu_page.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SHEET_GAP = 8  # px between the panels of a contact sheet
MIN_SPREAD = 1.0  # a carry-over needs references that vary by this much (L* or b* units, degrees of slant)


class EvalCrop(BaseModel):
    """A held-out crop as diffu_page.data.build_crops (or steer_set / style_pairs) wrote it."""

    file_name: str
    caption: str
    width: int
    height: int
    lines: list[LineRecord]
    reference: str | None = None
    reference_lines: list[list[int]] | None = None  # the reference's line boxes, to measure its hand
    seed: int | None = None
    variant: str | None = None

    def plan(self) -> CropPlan:
        """The crop as a plan: its stored caption and boxes, lines read top to bottom as numbered."""
        return CropPlan(
            rect=(0, 0, self.width, self.height),
            width=self.width,
            height=self.height,
            caption=self.caption,
            lines=self.lines,
            reading_order=list(range(len(self.lines))),
        )


class LineResult(BaseModel):
    """One line of a crop: its text, what the recognizer read in the generated and the real crop, and the CERs."""

    text: str
    gen: str
    gen_cer: float
    real: str
    real_cer: float


class Hands(BaseModel):
    """A drawn crop's hand beside its real crop's, both measured in the crop's line boxes, and beside its
    reference's, measured in the reference's at the reference size; and how far apart they are."""

    drawn: HandLook
    real: HandLook
    reference: HandLook | None = None
    to_real: HandDistance
    to_reference: HandDistance | None = None
    real_to_reference: HandDistance | None = None  # the pair's own gap: how alike the two real hands are


def crop_hands(
    drawn: Image.Image,
    real: Image.Image,
    lines: Sequence[LineRecord],
    reference: Image.Image | None = None,
    reference_lines: Sequence[Box] | None = None,
) -> Hands:
    """The hands of a drawn crop, its real crop and, given its line boxes, its reference."""
    boxes = [line.bbox for line in lines]
    drawn_hand, real_hand = hand_look(drawn, boxes), hand_look(real, boxes)
    if reference is None or not reference_lines:
        return Hands(drawn=drawn_hand, real=real_hand, to_real=hand_distance(drawn_hand, real_hand))
    reference_hand = hand_look(as_reference(reference), reference_lines)
    return Hands(
        drawn=drawn_hand,
        real=real_hand,
        reference=reference_hand,
        to_real=hand_distance(drawn_hand, real_hand),
        to_reference=hand_distance(drawn_hand, reference_hand),
        real_to_reference=hand_distance(real_hand, reference_hand),
    )


class CarryOver(BaseModel):
    """How much of the references' variation reaches the hands across the crops: the least-squares slope of a hand's
    measure on its reference's (1: follows the reference fully; 0: ignores it). Colours by CIE L*a*b* lightness L* and
    yellowness b* (cream and brown against grey and black). None where fewer than three crops have the measure or the
    references vary by less than ``MIN_SPREAD`` (the yellowness of grey inks is 0 plus rounding).

    Unlike a median distance, a slope is not fooled by a model's default: ink drawn near-black whatever the reference
    lands close to every black-inked reference, but its slope stays at 0."""

    paper_lightness: float | None
    paper_yellowness: float | None
    ink_lightness: float | None
    ink_yellowness: float | None
    slant: float | None


def _slope(pairs: Sequence[tuple[float, float]]) -> float | None:
    references = [x for x, _ in pairs]
    if len(pairs) < 3 or pstdev(references) < MIN_SPREAD:
        return None
    return linear_regression(references, [y for _, y in pairs]).slope


def carry_over(pairs: Sequence[tuple[HandLook, HandLook]]) -> CarryOver:
    """How much of the references' variation reaches the hands in ``pairs`` of (reference, hand)."""

    def colour(field: str, axis: int) -> float | None:
        measured = [(getattr(reference, field), getattr(hand, field)) for reference, hand in pairs]
        return _slope([(lab(a)[axis], lab(b)[axis]) for a, b in measured if a is not None and b is not None])

    slants = [
        (reference.slant, hand.slant)
        for reference, hand in pairs
        if reference.slant is not None and hand.slant is not None
    ]
    return CarryOver(
        paper_lightness=colour("paper", 0),
        paper_yellowness=colour("paper", 2),
        ink_lightness=colour("ink", 0),
        ink_yellowness=colour("ink", 2),
        slant=_slope(slants),
    )


class HandFidelity(BaseModel):
    """Medians over the drawn crops of each ``HandDistance`` measure: a drawn hand's distance to its real crop's;
    over the crops with a measured reference, its distance to its own reference's, to the other crops' references
    (chance: no lower than this when the reference's hand is ignored), and the real crop's distance to its
    reference's. And the carry-over from the references to the drawn hands, beside the carry-over to the real crops
    (what the references can tell of the real hands: the ceiling)."""

    to_real: HandDistance
    to_reference: HandDistance | None = None
    to_other_references: HandDistance | None = None
    real_to_reference: HandDistance | None = None
    carry_over: CarryOver | None = None
    real_carry_over: CarryOver | None = None


def _median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return median(present) if present else None


def _medians(distances: list[HandDistance]) -> HandDistance:
    return HandDistance.model_validate(
        {field: _median([getattr(distance, field) for distance in distances]) for field in HandDistance.model_fields}
    )


def hand_fidelity(hands: Sequence[Hands]) -> HandFidelity:
    """The medians of ``hands`` (at least one)."""
    fidelity = HandFidelity(to_real=_medians([hand.to_real for hand in hands]))
    referenced = [hand for hand in hands if hand.reference is not None]
    if not referenced:
        return fidelity
    others = [
        hand_distance(hand.drawn, other.reference)
        for hand in referenced
        for other in referenced
        if other is not hand and other.reference is not None
    ]
    return fidelity.model_copy(
        update={
            "to_reference": _medians([hand.to_reference for hand in referenced if hand.to_reference is not None]),
            "to_other_references": _medians(others) if others else None,
            "real_to_reference": _medians(
                [hand.real_to_reference for hand in referenced if hand.real_to_reference is not None]
            ),
            "carry_over": carry_over([(hand.reference, hand.drawn) for hand in referenced if hand.reference]),
            "real_carry_over": carry_over([(hand.reference, hand.real) for hand in referenced if hand.reference]),
        }
    )


class CropResult(BaseModel):
    """One evaluated crop, by its index in the eval set."""

    index: int
    file: str
    variant: str | None = None
    gen_cer: float
    real_cer: float
    lines: list[LineResult]
    hands: Hands | None = None  # None in a results.json written before hands were measured


class Summary(BaseModel):
    """The run's mean CERs over the drawn crops, how closely their hands follow (``HandFidelity``), and what drew
    them: the model, its LoRAs, the seed and the draw settings. Results written before the model, seed and settings
    were recorded load with None there."""

    n: int
    failed: int
    gen_cer: float
    real_cer: float
    hands: HandFidelity | None = None
    model: str | None = None
    loras: dict[str, str]
    seed: int | None = None
    settings: DrawSettings | None = None


class Results(BaseModel):
    """``results.json``: the summary, every drawn crop, and every crop that failed (index -> error)."""

    summary: Summary
    samples: list[CropResult]
    failed: dict[int, str]


def contact_sheet(panels: list[Image.Image], lines: list[LineRecord]) -> Image.Image:
    """The panels side by side at the last one's height ([reference,] real, generated), with the line boxes drawn
    on the generated crop."""
    generated = panels[-1]
    scaled = [
        panel.resize((round(panel.width * generated.height / panel.height), generated.height)) for panel in panels
    ]
    sheet = Image.new("RGB", (sum(p.width for p in scaled) + SHEET_GAP * (len(scaled) - 1), generated.height), "white")
    offset = 0
    for panel in scaled:
        sheet.paste(panel, (offset, 0))
        offset += panel.width + SHEET_GAP
    offset -= generated.width + SHEET_GAP
    draw = ImageDraw.Draw(sheet)
    for line in lines:
        x0, y0, x1, y1 = pixel_box(line.bbox, generated.size)
        draw.rectangle([offset + x0, y0, offset + x1, y1], outline=(255, 0, 0))
    return sheet


def read_crops(crops_dir: Path, n: int) -> list[EvalCrop]:
    """The first ``n`` crops of a split dir.

    Raises:
        ValueError: If some but not all of them name a reference image.
    """
    with (crops_dir / "metadata.jsonl").open(encoding="utf-8") as metadata:
        crops = [EvalCrop.model_validate_json(line) for line in metadata][:n]
    with_reference = sum(crop.reference is not None for crop in crops)
    if with_reference not in (0, len(crops)):
        raise ValueError(f"{with_reference} of {len(crops)} crops name a reference image: all or none must")
    return crops


def evaluate_crop(
    pipe: ModularPipeline,
    crop: EvalCrop,
    *,
    index: int,
    seed: int,
    crops_dir: Path,
    out: Path,
    settings: DrawSettings,
) -> CropResult:
    """Draw one held-out crop with the crop pipeline, read it and its real counterpart back, and save its contact
    sheet."""
    plan = crop.plan()
    reference = load_rgb(crops_dir / crop.reference) if crop.reference else None
    drawn = draw_crop(pipe, plan, name=f"{index:03d}", seed=seed, reference=reference, settings=settings)
    drawn.image.save(out / "generated" / f"{index:03d}.png")
    real = load_rgb(crops_dir / crop.file_name).resize(drawn.image.size)
    real_reads, real_cers = read_lines(real, plan.lines, model=pipe.htr_model, processor=pipe.htr_processor)
    panels = [reference, real, drawn.image] if reference else [real, drawn.image]
    contact_sheet(panels, crop.lines).save(out / f"{index:03d}_{Path(crop.file_name).stem}.jpg")
    lines = zip(crop.lines, drawn.line_reads, drawn.line_cers, real_reads, real_cers, strict=True)
    return CropResult(
        index=index,
        file=crop.file_name,
        variant=crop.variant,
        gen_cer=fmean(drawn.line_cers),
        real_cer=fmean(real_cers),
        lines=[LineResult(text=ln.text, gen=g, gen_cer=gc, real=r, real_cer=rc) for ln, g, gc, r, rc in lines],
        hands=crop_hands(drawn.image, real, crop.lines, reference, crop.reference_lines),
    )


@app.command()
def evaluate(
    crops: Annotated[
        Path, typer.Option(help="a split dir of diffu_page.data.build_crops (or steer_set / style_pairs)")
    ],
    out: Annotated[Path, typer.Option(help="where the sheets, generated/ and results.json go")],
    model: Annotated[
        Path | None, typer.Option(help="the model directory (FLUX.3 Klein or diffu); default $FLUX3_MODEL")
    ] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    n: Annotated[int, typer.Option("-n", help="crops to evaluate")] = 16,
    steps: Steps = DEFAULTS.steps,
    guidance: Guidance = DEFAULTS.guidance_scale,
    reference_guidance: ReferenceGuidance = DEFAULTS.reference_guidance_scale,
    seed: int = 0,
    text_layer: TextLayers = None,
    device: Device = DEFAULTS.device,
    max_line_cer: MaxLineCer = DEFAULTS.max_line_cer,
) -> None:
    """Draw held-out crops from their stored captions and compare their read-back with the real crops'."""
    model = required_model(model)
    eval_crops = read_crops(crops, n)
    if reference_guidance != 1.0 and not any(crop.reference for crop in eval_crops):
        raise typer.BadParameter(
            "these crops name no reference image to guide toward", param_hint="--reference-guidance"
        )
    (out / "generated").mkdir(parents=True, exist_ok=True)
    loras = {adapter.name: adapter.path for adapter in lora or []}
    draw_settings = DrawSettings(
        steps=steps,
        guidance_scale=guidance,
        reference_guidance_scale=reference_guidance,
        max_line_cer=max_line_cer,
        text_layers=tuple(text_layer) if text_layer else None,
        device=device,
    )
    pipe = load_pipeline(model, crop_blocks, loras=loras, settings=draw_settings)
    samples: list[CropResult] = []
    failed: dict[int, str] = {}
    for k, crop in enumerate(eval_crops):
        crop_seed = crop.seed if crop.seed is not None else seed + k
        try:
            sample = evaluate_crop(
                pipe, crop, index=k, seed=crop_seed, crops_dir=crops, out=out, settings=draw_settings
            )
        except Exception as error:  # one crop must not end a long evaluation; results.json records it
            failed[k] = f"{type(error).__name__}: {error}"
            err_console.print(f"{k:3d} {crop.file_name} failed: {failed[k]}")
            continue
        samples.append(sample)
        console.print(f"{k:3d} {crop.file_name}: gen CER {sample.gen_cer:.3f}, real CER {sample.real_cer:.3f}")
    if not samples:
        err_console.print("no crop could be evaluated")
        raise typer.Exit(code=1)
    summary = Summary(
        n=len(samples),
        failed=len(failed),
        gen_cer=fmean(s.gen_cer for s in samples),
        real_cer=fmean(s.real_cer for s in samples),
        hands=hand_fidelity(measured) if (measured := [s.hands for s in samples if s.hands is not None]) else None,
        model=str(model),
        loras={name: str(path) for name, path in loras.items()},
        seed=seed,
        settings=draw_settings,
    )
    (out / "results.json").write_text(
        Results(summary=summary, samples=samples, failed=failed).model_dump_json(indent=1)
    )
    print(summary.model_dump_json())  # machine output, one line: rich would wrap it at the terminal's width
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
