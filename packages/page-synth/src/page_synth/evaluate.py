"""Evaluate FLUX.3 (+ LoRAs) on held-out crops: draw each from its stored caption and read every line back.

Every crop goes through the same :class:`page_synth.generator.CropGenerator` as page generation; a crop that names a
``reference`` image is drawn conditioned on it (v3, style by example). A crop's optional ``seed`` overrides
``--seed + index`` (the steering test draws every variant of a template from the same noise). Writes contact sheets
(``[reference,] real, generated``), ``generated/NNN.png`` and ``results.json``.

    python -m page_synth.evaluate --crops data_flux3/layout_crops_v2/val_eval --lora v2=<ckpt>/model --out <dir> -n 24
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from pydantic import BaseModel

from page_synth.caption import LineRecord
from page_synth.generate import lora_arg
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.pipelines import FLUX3_KLEIN
from page_synth.plan import CropPlan
from page_synth.readback import mean, pixel_box

SHEET_GAP = 8  # px between the panels of a contact sheet


class EvalCrop(BaseModel):
    """A held-out crop as page_synth.data.build_crops (or steer_set / style_pairs) wrote it."""

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
    text: str
    gen: str
    gen_cer: float
    real: str
    real_cer: float


class CropResult(BaseModel):
    file: str
    variant: str | None = None
    gen_cer: float
    real_cer: float
    lines: list[LineResult]


class Summary(BaseModel):
    n: int
    gen_cer: float
    real_cer: float
    loras: dict[str, str]
    guidance: float
    steps: int


class Results(BaseModel):
    summary: Summary
    samples: list[CropResult]


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
    """The first ``n`` crops of a split dir; all or none of them name a reference image."""
    with (crops_dir / "metadata.jsonl").open(encoding="utf-8") as metadata:
        crops = [EvalCrop.model_validate_json(line) for line in metadata][:n]
    with_reference = sum(crop.reference is not None for crop in crops)
    if with_reference not in (0, len(crops)):
        raise ValueError(f"{with_reference} of {len(crops)} crops name a reference image: all or none must")
    return crops


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, type=Path, help="a split dir of page_synth.data.build_crops")
    ap.add_argument("--model", type=Path, default=FLUX3_KLEIN)
    ap.add_argument("--lora", type=lora_arg, action="append", default=[], help="name=<NeMo LoRA model dir>")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-n", type=int, default=16, help="crops to generate")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    crops = read_crops(args.crops, args.n)
    (args.out / "generated").mkdir(parents=True, exist_ok=True)
    loras = dict(args.lora)
    settings = GeneratorSettings(steps=args.steps, guidance_scale=args.guidance)
    generate = CropGenerator(args.model, loras=loras, settings=settings)

    samples: list[CropResult] = []
    for k, crop in enumerate(crops):
        plan = crop.plan()
        reference = Image.open(args.crops / crop.reference).convert("RGB") if crop.reference else None
        seed = crop.seed if crop.seed is not None else args.seed + k
        drawn = generate(plan, seed=seed, image_name=f"{k:03d}.png", reference=reference)
        drawn.image.save(args.out / "generated" / f"{k:03d}.png")
        real = Image.open(args.crops / crop.file_name).convert("RGB").resize(drawn.image.size)
        real_reads, real_cers = generate.read(real, plan)
        panels = [reference, real, drawn.image] if reference else [real, drawn.image]
        contact_sheet(panels, crop.lines).save(args.out / f"{k:03d}_{Path(crop.file_name).stem}.jpg")
        lines = zip(crop.lines, drawn.line_reads, drawn.line_cers, real_reads, real_cers, strict=True)
        sample = CropResult(
            file=crop.file_name,
            variant=crop.variant,
            gen_cer=mean(drawn.line_cers),
            real_cer=mean(real_cers),
            lines=[LineResult(text=ln.text, gen=g, gen_cer=gc, real=r, real_cer=rc) for ln, g, gc, r, rc in lines],
        )
        samples.append(sample)
        print(f"{k:3d} {sample.file}: gen_CER {sample.gen_cer:.3f}  real_CER {sample.real_cer:.3f}", flush=True)

    summary = Summary(
        n=len(samples),
        gen_cer=mean([s.gen_cer for s in samples]),
        real_cer=mean([s.real_cer for s in samples]),
        loras={name: str(path) for name, path in loras.items()},
        guidance=args.guidance,
        steps=args.steps,
    )
    (args.out / "results.json").write_text(Results(summary=summary, samples=samples).model_dump_json(indent=1))
    print(summary.model_dump_json())


if __name__ == "__main__":
    main()
