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
    MaxLineCer,
    Model,
    ReferenceGuidanceScale,
    Steps,
    TextLayers,
    lora_dirs,
    required_model,
    text_layers,
)
from diffu_page.crops.geometry import pixel_box
from diffu_page.evalset import EvalCrop, generated_path, read_crops
from diffu_page.fidelity import HandFidelity, Hands, crop_hands, hand_fidelity
from diffu_page.output import JPEG_QUALITY, console, err_console
from diffu_page.pipeline import DEVICE_ERRORS, DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.readback import read_lines
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


class CropResult(BaseModel):
    """One evaluated crop, by its index in the eval set."""

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
    reference = crop.reference_image(crops_dir)
    drawn = draw_crop(pipe, plan, name=f"{index:03d}", seed=seed, reference=reference, settings=settings)
    drawn.image.save(generated_path(out, index))
    real = crop.real_image(crops_dir, drawn.image.size)
    real_reads, real_cers = read_lines(real, plan.lines, model=pipe.htr_model, processor=pipe.htr_processor)
    panels = [reference, real, drawn.image] if reference else [real, drawn.image]
    contact_sheet(panels, crop.lines).save(out / f"{index:03d}_{Path(crop.file_name).stem}.jpg", quality=JPEG_QUALITY)
    lines = zip(crop.lines, drawn.line_reads, drawn.line_cers, real_reads, real_cers, strict=True)
    return CropResult(
        index=index,
        file=crop.file_name,
        variant=crop.variant,
        gen_cer=fmean(drawn.line_cers),
        real_cer=fmean(real_cers),
        lines=[LineResult(text=ln.text, gen=g, gen_cer=gc, real=r, real_cer=rc) for ln, g, gc, r, rc in lines],
        hands=crop_hands(drawn.image, real, crop.lines, reference=reference, reference_lines=crop.reference_lines),
    )


def evaluate_crops(
    pipe: ModularPipeline, eval_crops: list[EvalCrop], *, crops: Path, out: Path, seed: int, settings: DrawSettings
) -> tuple[list[CropResult], dict[int, str]]:
    """Every crop evaluated (:func:`evaluate_crop`, seeded by its own seed or ``seed`` + its index); a crop that fails
    is recorded by its index with its error and the rest still run, unless the device failed: then every crop after
    it is recorded as not drawn."""
    samples: list[CropResult] = []
    failed: dict[int, str] = {}
    for k, crop in enumerate(eval_crops):
        crop_seed = crop.seed if crop.seed is not None else seed + k
        try:
            sample = evaluate_crop(pipe, crop, index=k, seed=crop_seed, crops_dir=crops, out=out, settings=settings)
        except DEVICE_ERRORS as error:
            failed[k] = f"{type(error).__name__}: {error}"
            failed |= dict.fromkeys(range(k + 1, len(eval_crops)), "not drawn: the device failed on an earlier crop")
            err_console.print(f"{k:3d} {crop.file_name} failed on the device, the evaluation stops: {failed[k]}")
            break
        except Exception as error:  # one crop must not end a long evaluation; results.json records it
            failed[k] = f"{type(error).__name__}: {error}"
            err_console.print(f"{k:3d} {crop.file_name} failed: {failed[k]}")
            continue
        samples.append(sample)
        console.print(f"{k:3d} {crop.file_name}: gen CER {sample.gen_cer:.3f}, real CER {sample.real_cer:.3f}")
    return samples, failed


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
    fuse_loras: FuseLoras = DEFAULTS.fuse_loras,
    compile_blocks: CompileBlocks = DEFAULTS.compile_blocks,
) -> None:
    """Draw held-out crops from their stored captions and compare their read-back with the real crops'."""
    model = required_model(model)
    eval_crops = read_crops(crops, n)
    draw_settings = DrawSettings(
        steps=steps,
        guidance_scale=guidance,
        reference_guidance_scale=reference_guidance,
        guidance_stop=guidance_stop,
        max_line_cer=max_line_cer,
        text_layers=text_layers(text_layer),
        device=device,
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
    samples, failed = evaluate_crops(pipe, eval_crops, crops=crops, out=out, seed=seed, settings=draw_settings)
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
        Results(summary=summary, samples=samples, failed=failed).model_dump_json(indent=1)
    )
    print(summary.model_dump_json())  # machine output, one line: rich would wrap it at the terminal's width
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
