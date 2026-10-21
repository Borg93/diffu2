"""Evaluate FLUX.3 Klein (+ a LoRA) on held-out crops: generate each from its stored caption and read every line back.

Built on Modular Diffusers: FLUX.3's ``text2image`` workflow (``image_conditioned`` when the crops name a ``reference``
image) with its prompt upsampler popped (a crop's stored training caption is encoded as given) and
:class:`page_synth.blocks.LineReadbackStep` appended. A crop's optional ``seed`` overrides ``--seed + index`` (the
steering test draws every variant of a template from the same noise). Writes contact sheets
(``[reference,] real, generated``), ``generated/NNN.png`` and ``results.json``.

    python -m page_synth.evaluate --crops data_flux3/layout_crops_v2/val_eval --lora <ckpt>/model --out <dir> -n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import Flux3AutoBlocks, ModularPipeline
from PIL import Image, ImageDraw
from pydantic import BaseModel
from transformers import TrOCRProcessor

from page_synth.blocks import (
    DEFAULT_MODEL,
    HTR,
    LineReadbackStep,
    apply_nemo_lora,
    load_trocr,
    mean,
    pixel_box,
    read_lines,
)

SHEET_GAP = 8  # px between the real and the generated crop on a contact sheet


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
    lora: str | None
    guidance: float
    steps: int


class Results(BaseModel):
    summary: Summary
    samples: list[CropResult]


def build_pipeline(
    model_path: str, guidance_scale: float, workflow: Literal["text2image", "image_conditioned"]
) -> ModularPipeline:
    """A FLUX.3 workflow without the prompt upsampler (it would rewrite the quoted lines), then the read-back."""
    blocks = Flux3AutoBlocks().get_workflow(workflow)
    blocks.sub_blocks.pop("prompt_upsampler")
    blocks.sub_blocks.insert("line_readback", LineReadbackStep(), len(blocks.sub_blocks))
    pipe = blocks.init_pipeline(model_path)
    pipe.load_components(
        names=[name for name in pipe.pretrained_component_names if not name.startswith("htr_")],
        dtype={"vae": torch.float32, "default": torch.bfloat16},
    )
    pipe.update_components(htr_model=load_trocr(), htr_processor=TrOCRProcessor.from_pretrained(HTR))
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance_scale))
    pipe.to("cuda")
    return pipe


def contact_sheet(panels: list[Image.Image], lines: list[dict]) -> Image.Image:
    """The panels side by side at the last one's height ([reference,] real, generated), with the line boxes drawn
    on the generated crop."""
    generated = panels[-1]
    scaled = [
        panel.resize((round(panel.width * generated.height / panel.height), generated.height))
        for panel in panels
    ]
    sheet = Image.new(
        "RGB", (sum(p.width for p in scaled) + SHEET_GAP * (len(scaled) - 1), generated.height), "white"
    )
    offset = 0
    for panel in scaled:
        sheet.paste(panel, (offset, 0))
        offset += panel.width + SHEET_GAP
    offset -= generated.width + SHEET_GAP
    draw = ImageDraw.Draw(sheet)
    for line in lines:
        x0, y0, x1, y1 = pixel_box(line["bbox"], generated.size)
        draw.rectangle([offset + x0, y0, offset + x1, y1], outline=(255, 0, 0))
    return sheet


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, type=Path, help="a split dir of page_synth.data.build_crops")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--lora", default=None, help="NeMo LoRA checkpoint model/ dir; omit for the base model")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-n", type=int, default=16, help="crops to generate")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    with (args.crops / "metadata.jsonl").open(encoding="utf-8") as metadata:
        rows = [json.loads(line) for line in metadata][: args.n]
    with_reference = sum("reference" in row for row in rows)
    if with_reference not in (0, len(rows)):
        raise SystemExit(f"{with_reference} of {len(rows)} crops name a reference image: all or none must")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "generated").mkdir(exist_ok=True)
    pipe = build_pipeline(args.model, args.guidance, "image_conditioned" if with_reference else "text2image")
    if args.lora:
        print(
            f"LoRA: {apply_nemo_lora(pipe.transformer, Path(args.lora))} modules from {args.lora}", flush=True
        )

    samples: list[CropResult] = []
    for k, row in enumerate(rows):
        reference = Image.open(args.crops / row["reference"]).convert("RGB") if with_reference else None
        state = pipe(
            prompt=row["caption"],  # the crop's training caption, as the LoRA saw it
            page_lines=row["lines"],
            height=round(row["height"] / 16) * 16,
            width=round(row["width"] / 16) * 16,
            num_inference_steps=args.steps,
            generator=torch.Generator().manual_seed(row.get("seed", args.seed + k)),
            output=["images", "line_reads", "line_cers"],
            **({"image": reference} if reference else {}),
        )
        generated = state["images"][0]
        generated.save(args.out / "generated" / f"{k:03d}.png")
        real = Image.open(args.crops / row["file_name"]).convert("RGB").resize(generated.size)
        real_reads, real_cers = read_lines(
            real, row["lines"], model=pipe.htr_model, processor=pipe.htr_processor
        )
        panels = [reference, real, generated] if reference else [real, generated]
        contact_sheet(panels, row["lines"]).save(args.out / f"{k:03d}_{Path(row['file_name']).stem}.jpg")
        lines = zip(row["lines"], state["line_reads"], state["line_cers"], real_reads, real_cers, strict=True)
        sample = CropResult(
            file=row["file_name"],
            variant=row.get("variant"),
            gen_cer=mean(state["line_cers"]),
            real_cer=mean(real_cers),
            lines=[
                LineResult(text=ln["text"], gen=g, gen_cer=gc, real=r, real_cer=rc)
                for ln, g, gc, r, rc in lines
            ],
        )
        samples.append(sample)
        print(
            f"{k:3d} {sample.file}: gen_CER {sample.gen_cer:.3f}  real_CER {sample.real_cer:.3f}", flush=True
        )

    summary = Summary(
        n=len(samples),
        gen_cer=mean([s.gen_cer for s in samples]),
        real_cer=mean([s.real_cer for s in samples]),
        lora=args.lora,
        guidance=args.guidance,
        steps=args.steps,
    )
    (args.out / "results.json").write_text(
        Results(summary=summary, samples=samples).model_dump_json(indent=1)
    )
    print(summary.model_dump_json())


if __name__ == "__main__":
    main()
