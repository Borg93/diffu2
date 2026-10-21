"""Generate held-out page crops from their layout captions with FLUX.3 Klein (+ a LoRA) and read every line back.

Built on Modular Diffusers: the FLUX.3 ``text2image`` workflow (``image_conditioned`` when the crops name a
``reference`` image, the v3 reference-style LoRA) with its prompt upsampler popped (a crop's stored training caption
is encoded as given) and ``LineReadbackStep`` appended, which crops each line's box out of the generated image, reads
it with a Swedish HTR TrOCR and returns the per-line character error rate. A crop's optional ``seed`` overrides
``--seed + index`` (the steering test draws every variant of a template from the same noise).

Runs in the NeMo AutoModel environment (diffusers main with FLUX.3); the LoRA is NeMo's adapter format, applied
to the pipeline's transformer the way NeMo's ``examples/diffusion/generate/generate.py`` does.

    cd ~/Desktop/repos/Automodel && uv run --no-sync python ~/Desktop/repos/diffu2/experiments/flux3_klein/eval_layout.py \
        --crops ~/Desktop/repos/diffu2/data_flux3/layout_crops_v2/val_eval --lora <ckpt>/model --out <dir> -n 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import torch
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.modular_pipelines import (
    ComponentSpec,
    Flux3AutoBlocks,
    InputParam,
    ModularPipeline,
    ModularPipelineBlocks,
    OutputParam,
    PipelineState,
)
from PIL import Image, ImageDraw
from pydantic import BaseModel
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

HTR = "Riksarkivet/trocr-base-handwritten-hist-swe-2"
DEFAULT_MODEL = Path.home() / "Desktop/repos/flux3-models/flux-3-image-klein-base"
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


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    return levenshtein(pred, ref) / max(len(ref), 1)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def pixel_box(bbox: list[int], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """A layout box (``[y_min, x_min, y_max, x_max]`` on 0-1000) as ``(x0, y0, x1, y1)`` pixels of ``size``."""
    width, height = size
    y0, x0, y1, x1 = bbox
    return x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000


def read_lines(
    image: Image.Image, lines: list[dict], *, model: VisionEncoderDecoderModel, processor: TrOCRProcessor
) -> tuple[list[str], list[float]]:
    """Read each line's box of ``image`` with TrOCR: the reads and their CERs against the lines' texts."""
    crops = [image.crop(pixel_box(line["bbox"], image.size)).convert("RGB") for line in lines]
    pixels = processor(images=crops, return_tensors="pt").pixel_values
    ids = model.generate(pixels.to(model.device, model.dtype), max_new_tokens=128)
    reads = [read.strip() for read in processor.batch_decode(ids, skip_special_tokens=True)]
    return reads, [cer(read, line["text"]) for read, line in zip(reads, lines, strict=True)]


class LineReadbackStep(ModularPipelineBlocks):
    """Reads every line box of the generated image with an HTR model: the per-line CER against its text."""

    model_name = "flux3"

    @property
    def description(self) -> str:
        return "Crops each page line's box from the generated image and reads it with TrOCR (per-line CER)."

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("htr_model", VisionEncoderDecoderModel, pretrained_model_name_or_path=HTR),
            ComponentSpec("htr_processor", TrOCRProcessor, pretrained_model_name_or_path=HTR),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("images", required=True, description="The generated images (PIL)."),
            InputParam(
                "page_lines",
                type_hint=list,
                required=True,
                description="[{'text', 'bbox' (0-1000, y first)}]",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "line_reads", type_hint=list, description="What the recognizer read in each line box."
            ),
            OutputParam("line_cers", type_hint=list, description="Per-line CER against the line's text."),
        ]

    @torch.no_grad()
    def __call__(self, components, state: PipelineState):
        block_state = self.get_block_state(state)
        block_state.line_reads, block_state.line_cers = read_lines(
            block_state.images[0],
            block_state.page_lines,
            model=components.htr_model,
            processor=components.htr_processor,
        )
        self.set_block_state(state, block_state)
        return components, state


def apply_nemo_lora(transformer: torch.nn.Module, lora_dir: Path) -> int:
    """Apply a NeMo AutoModel LoRA checkpoint (``adapter_config.json`` + ``adapter_model.safetensors``).

    Raises:
        RuntimeError: If no module took the adapter or its LoRA keys don't match the transformer.
    """
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from safetensors.torch import load_file

    config = json.loads((lora_dir / "adapter_config.json").read_text())
    for ours, peft_name in (("dim", "r"), ("alpha", "lora_alpha"), ("dropout", "lora_dropout")):
        config.setdefault(ours, config.get(peft_name))
    patched = apply_lora_to_linear_modules(transformer, PeftConfig.from_dict(config), skip_freeze=True)
    weights = load_file(lora_dir / "adapter_model.safetensors")
    result = transformer.load_state_dict(
        {k.removeprefix("base_model.model."): v for k, v in weights.items()}, strict=False
    )
    mismatched = [k for k in result.missing_keys + result.unexpected_keys if ".lora_" in k]
    if patched == 0 or mismatched:
        raise RuntimeError(f"LoRA did not load cleanly: {patched} modules, mismatched keys {mismatched[:5]}")
    return patched


def load_trocr() -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(HTR, dtype=torch.bfloat16, device_map="cuda")
    # TrOCR builds its sinusoidal position table as a plain attribute in __init__, so a meta-device load leaves
    # it on meta ("Cannot copy out of meta tensor"): rebuild it on the model's device.
    for module in model.modules():
        if type(module).__name__ == "TrOCRSinusoidalPositionalEmbedding" and module.weights.is_meta:
            table = module.get_embedding(module.weights.shape[0], module.embedding_dim, module.padding_idx)
            module.weights = table.to(model.device, model.dtype)
    return model


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
    ap.add_argument("--crops", required=True, type=Path, help="a split dir of build_layout_crops.py")
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
