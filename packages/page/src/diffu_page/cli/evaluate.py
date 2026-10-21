"""Evaluate a model (FLUX.3 + LoRAs, or diffu) on held-out crops: draw each from its stored caption with the crop
pipeline (:func:`diffu_page.pipeline.crop_blocks`) and read every line back, beside the real crop's read-back.

A crop that names a ``reference`` image is drawn conditioned on it (v3, style by example). A crop's optional ``seed``
overrides ``--seed + index`` (the steering test draws every variant of a template from the same noise). Writes contact
sheets (``[reference,] real, generated``), ``generated/NNN.png`` and ``results.json``; a crop that fails is recorded
there and the rest still run.

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
from diffu_page.cli.options import Lora, parse_lora
from diffu_page.output import console, err_console
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.plan import CropPlan
from diffu_page.readback import pixel_box, read_lines
from diffu_page.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

SHEET_GAP = 8  # px between the panels of a contact sheet


class EvalCrop(BaseModel):
    """A held-out crop as diffu_page.data.build_crops (or steer_set / style_pairs) wrote it."""

    file_name: str
    caption: str
    width: int
    height: int
    lines: list[LineRecord]
    reference: str | None = None
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


class CropResult(BaseModel):
    """One evaluated crop, by its index in the eval set."""

    index: int
    file: str
    variant: str | None = None
    gen_cer: float
    real_cer: float
    lines: list[LineResult]


class Summary(BaseModel):
    """The run's mean CERs over the drawn crops, and its settings."""

    n: int
    failed: int
    gen_cer: float
    real_cer: float
    loras: dict[str, str]
    guidance: float
    steps: int


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


def _load(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


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
    reference = _load(crops_dir / crop.reference) if crop.reference else None
    drawn = draw_crop(pipe, plan, name=f"{index:03d}", seed=seed, reference=reference, settings=settings)
    drawn.image.save(out / "generated" / f"{index:03d}.png")
    real = _load(crops_dir / crop.file_name).resize(drawn.image.size)
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
    )


@app.command()
def evaluate(
    crops: Annotated[
        Path, typer.Option(help="a split dir of diffu_page.data.build_crops (or steer_set / style_pairs)")
    ],
    out: Annotated[Path, typer.Option(help="where the sheets, generated/ and results.json go")],
    model: Annotated[Path, typer.Option(help="the model directory (FLUX.3 Klein or diffu)")] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    n: Annotated[int, typer.Option("-n", help="crops to evaluate")] = 16,
    steps: int = 50,
    guidance: float = 4.0,
    seed: int = 0,
    text_layer: Annotated[
        list[int] | None, typer.Option(help="a text-encoder layer to stack (repeat it; default: the model's own)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the models run")] = "cuda",
) -> None:
    """Draw held-out crops from their stored captions and compare their read-back with the real crops'."""
    eval_crops = read_crops(crops, n)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    loras = {adapter.name: adapter.path for adapter in lora or []}
    draw_settings = DrawSettings(
        steps=steps, guidance_scale=guidance, text_layers=tuple(text_layer) if text_layer else None, device=device
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
        loras={name: str(path) for name, path in loras.items()},
        guidance=guidance,
        steps=steps,
    )
    (out / "results.json").write_text(
        Results(summary=summary, samples=samples, failed=failed).model_dump_json(indent=1)
    )
    console.print(summary.model_dump_json())
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
