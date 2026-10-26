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

from pathlib import Path
from statistics import fmean
from typing import Annotated

import typer
from diffusers.modular_pipelines import ModularPipeline
from PIL import Image, ImageDraw
from pydantic import BaseModel

from diffu_page.captions.prompt import LineRecord
from diffu_page.cli.options import (
    DEFAULTS,
    CompileBlocks,
    Device,
    FuseLoras,
    Guidance,
    GuidanceStop,
    Loras,
    LoraScale,
    MaxLineCer,
    Model,
    ReferenceGuidanceScale,
    SamplingShift,
    Steps,
    TextLayers,
    lora_dirs,
    required_model,
    settings_from_options,
)
from diffu_page.crops.geometry import pixel_box
from diffu_page.evalset import EvalCrop, generated_path, read_crops
from diffu_page.fidelity import HandFidelity, Hands, crop_hands, hand_fidelity
from diffu_page.judge import Judge, TrOCRJudge
from diffu_page.local_run import run_each
from diffu_page.output import JPEG_QUALITY, console, err_console, side_by_side
from diffu_page.pipeline import DEVICE_ERRORS, DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SHEET_GAP = 8  # px between the panels of a contact sheet


class LineResult(BaseModel):
    """One line of a crop: its text, what the recognizer read in the generated and the real crop, and the CERs."""

    text: str
    gen: str
    gen_cer: float
    real: str
    real_cer: float


class CropEvaluation(BaseModel):
    """One crop's evaluation as ``results.json`` lists it, by its index in the eval set: the drawn and the real crop
    read back, line by line, and their hands."""

    index: int
    file: str
    variant: str | None = None
    gen_cer: float
    real_cer: float
    lines: list[LineResult]
    hands: Hands


class Summary(BaseModel):
    """The run's mean CERs over the drawn crops, how closely their hands follow (``HandFidelity``), and what drew
    them: the model, its LoRAs, the seed and the draw settings."""

    n: int
    failed: int
    gen_cer: float
    real_cer: float
    hands: HandFidelity
    model: str
    loras: dict[str, str]
    seed: int
    settings: DrawSettings


class Results(BaseModel):
    """``results.json``: the summary, every drawn crop, and every crop that failed (index -> error)."""

    summary: Summary
    samples: list[CropEvaluation]
    failed: dict[int, str]


def contact_sheet(panels: list[Image.Image], lines: list[LineRecord]) -> Image.Image:
    """The panels side by side at the last one's height ([reference,] real, generated), with the line boxes drawn
    on the generated crop."""
    generated = panels[-1]
    sheet = side_by_side(panels, height=generated.height, gap=SHEET_GAP)
    offset = sheet.width - generated.width
    draw = ImageDraw.Draw(sheet)
    for line in lines:
        x0, y0, x1, y1 = pixel_box(line.bbox, generated.size)
        draw.rectangle([offset + x0, y0, offset + x1, y1], outline=(255, 0, 0))
    return sheet


def evaluate_crop(
    pipe: ModularPipeline,
    crop: EvalCrop,
    *,
    index: int,
    seed: int,
    crops_dir: Path,
    out: Path,
    settings: DrawSettings,
    judge: Judge,
) -> CropEvaluation:
    """Draw one held-out crop with the crop pipeline, read it and its real counterpart back, and save its contact
    sheet."""
    plan = crop.plan()
    reference = crop.reference_image(crops_dir)
    drawn = draw_crop(pipe, plan, name=f"{index:03d}", seed=seed, reference=reference, settings=settings, judge=judge)
    drawn.image.save(generated_path(out, index))
    real = crop.real_image(crops_dir, drawn.image.size)
    real_verdicts = judge(real, plan.lines)
    real_reads, real_cers = [v.read for v in real_verdicts], [v.cer for v in real_verdicts]
    panels = [reference, real, drawn.image] if reference else [real, drawn.image]
    contact_sheet(panels, crop.lines).save(out / f"{index:03d}_{Path(crop.file_name).stem}.jpg", quality=JPEG_QUALITY)
    lines = zip(crop.lines, drawn.line_reads, drawn.line_cers, real_reads, real_cers, strict=True)
    return CropEvaluation(
        index=index,
        file=crop.file_name,
        variant=crop.variant,
        gen_cer=fmean(drawn.line_cers),
        real_cer=fmean(real_cers),
        lines=[LineResult(text=ln.text, gen=g, gen_cer=gc, real=r, real_cer=rc) for ln, g, gc, r, rc in lines],
        hands=crop_hands(drawn.image, real, crop.lines, reference=reference, reference_lines=crop.reference_lines),
    )


def evaluate_crops(
    pipe: ModularPipeline,
    eval_crops: list[EvalCrop],
    *,
    crops: Path,
    out: Path,
    seed: int,
    settings: DrawSettings,
    judge: Judge,
) -> tuple[list[CropEvaluation], dict[int, str]]:
    """Every crop evaluated (:func:`evaluate_crop`, seeded by its own seed or ``seed`` + its index); a crop that fails
    is recorded by its index with its error and the rest still run, unless the device failed: then every crop after
    it is recorded as not run (:func:`diffu_page.local_run.run_each`)."""

    def evaluate_one(indexed: tuple[int, EvalCrop]) -> CropEvaluation:
        k, crop = indexed
        crop_seed = crop.seed if crop.seed is not None else seed + k
        sample = evaluate_crop(
            pipe, crop, index=k, seed=crop_seed, crops_dir=crops, out=out, settings=settings, judge=judge
        )
        console.print(f"{k:3d} {crop.file_name}: gen CER {sample.gen_cer:.3f}, real CER {sample.real_cer:.3f}")
        return sample

    return run_each(list(enumerate(eval_crops)), evaluate_one, key=lambda indexed: indexed[0], stop_on=DEVICE_ERRORS)


@app.command()
def evaluate(
    crops: Annotated[
        Path, typer.Option(help="a split dir of diffu_page.data.build_crops (or steer_set / style_pairs)")
    ],
    out: Annotated[Path, typer.Option(help="where the sheets, generated/ and results.json go")],
    model: Model = settings.flux3_model,
    lora: Loras = None,
    n: Annotated[int, typer.Option("-n", min=1, help="crops to evaluate")] = 16,
    steps: Steps = DEFAULTS.steps,
    guidance: Guidance = DEFAULTS.guidance_scale,
    reference_guidance: ReferenceGuidanceScale = DEFAULTS.reference_guidance_scale,
    guidance_stop: GuidanceStop = DEFAULTS.guidance_stop,
    seed: int = 0,
    text_layer: TextLayers = None,
    device: Device = DEFAULTS.device,
    max_line_cer: MaxLineCer = DEFAULTS.max_line_cer,
    lora_scale: LoraScale = DEFAULTS.lora_scale,
    sampling_shift: SamplingShift = DEFAULTS.sampling_shift,
    fuse_loras: FuseLoras = DEFAULTS.fuse_loras,
    compile_blocks: CompileBlocks = DEFAULTS.compile_blocks,
) -> None:
    """Draw held-out crops from their stored captions and compare their read-back with the real crops'."""
    model = required_model(model)
    eval_crops = read_crops(crops, n)
    draw_settings = settings_from_options(
        steps=steps,
        guidance=guidance,
        reference_guidance=reference_guidance,
        guidance_stop=guidance_stop,
        text_layer=text_layer,
        device=device,
        max_line_cer=max_line_cer,
        lora_scale=lora_scale,
        sampling_shift=sampling_shift,
        fuse_loras=fuse_loras,
        compile_blocks=compile_blocks,
    )
    if draw_settings.guides_reference and not any(crop.reference for crop in eval_crops):
        raise typer.BadParameter(
            "these crops name no reference image to guide toward", param_hint="--reference-guidance"
        )
    (out / "generated").mkdir(parents=True, exist_ok=True)
    loras = lora_dirs(lora)
    pipe = load_pipeline(model, crop_blocks, loras=loras, settings=draw_settings)
    judge = TrOCRJudge(device=device)
    samples, failed = evaluate_crops(
        pipe, eval_crops, crops=crops, out=out, seed=seed, settings=draw_settings, judge=judge
    )
    if not samples:
        err_console.print("no crop could be evaluated")
        raise typer.Exit(code=1)
    summary = Summary(
        n=len(samples),
        failed=len(failed),
        gen_cer=fmean(s.gen_cer for s in samples),
        real_cer=fmean(s.real_cer for s in samples),
        hands=hand_fidelity([s.hands for s in samples]),
        model=str(model),
        loras={name: str(path) for name, path in loras.items()},
        seed=seed,
        settings=draw_settings,
    )
    (out / "results.json").write_text(
        Results(summary=summary, samples=samples, failed=failed).model_dump_json(indent=1), encoding="utf-8"
    )
    print(summary.model_dump_json())  # machine output, one line: rich would wrap it at the terminal's width
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
