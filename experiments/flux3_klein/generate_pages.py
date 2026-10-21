"""Generate synthetic archive crops with ALTO/PAGE ground truth: any page template, any text, a chosen style.

For each template (a real PAGE/ALTO page, replayed with diffu-page) the running text is replaced by new text
(``refill``), the page is planned as training-scale crops (``page_plan``), and every crop goes through the
generator (``page_blocks``): FLUX.3 Klein + a LoRA draws it, TrOCR reads every line back, and its labels are
written with badly-read lines tagged ``unclear``. Writes ``<out>/<page>_<k>.jpg`` + ``.xml`` per crop and
``<out>/summary.json``. Runs in the NeMo AutoModel environment, one GPU job, through scripts/spark_guard.sh:

    cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/scripts/spark_guard.sh --need-gb 45 --name pages -- \\
        .venv-managed/bin/python ~/Desktop/repos/diffu2/experiments/flux3_klein/generate_pages.py \\
        --templates <page.xml ...> --text-from <page.xml ...> --lora <ckpt>/model --out <dir> \\
        --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from build_layout_crops import Provenance, ScanLook
from diffu_page import load_xml_layout, refill, running_text
from eval_layout import DEFAULT_MODEL, apply_nemo_lora, mean
from page_blocks import build_generator
from page_plan import PageStyle, plan_page
from pydantic import BaseModel


class CropResult(BaseModel):
    image: str
    lines: int
    unclear_lines: int
    cer: float


class PageResult(BaseModel):
    template: str
    crops: list[CropResult]


class Summary(BaseModel):
    style: PageStyle
    lora: str | None
    steps: int
    pages: list[PageResult]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--templates", type=Path, nargs="+", required=True, help="PAGE/ALTO pages whose layout to use"
    )
    ap.add_argument(
        "--text-from", type=Path, nargs="+", required=True, help="PAGE/ALTO pages whose text to write"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--lora", default=None, help="NeMo LoRA checkpoint model/ dir")
    ap.add_argument("--title", required=True, help="the archive, as the captions name it")
    ap.add_argument("--years", default="")
    ap.add_argument("--greyscale", action="store_true")
    ap.add_argument("--paper", default="cream")
    ap.add_argument("--ink", default="dark brown")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["alto", "page"], default="alto")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    style = PageStyle(
        provenance=Provenance(title=args.title, years=args.years),
        look=ScanLook(greyscale=args.greyscale, paper=args.paper, ink=args.ink),
    )
    text = " ".join(running_text(load_xml_layout(str(path), single_page=False)) for path in args.text_from)
    args.out.mkdir(parents=True, exist_ok=True)
    pipe = build_generator(args.model)
    if args.lora:
        print(
            f"LoRA: {apply_nemo_lora(pipe.transformer, Path(args.lora))} modules from {args.lora}", flush=True
        )

    pages: list[PageResult] = []
    for t, template in enumerate(args.templates):
        layout = refill(load_xml_layout(str(template), single_page=False), [text])
        crops: list[CropResult] = []
        for k, plan in enumerate(plan_page(layout, style)):
            name = f"{t:03d}_{template.stem}_{k}"
            state = pipe(
                crop_plan=plan,
                num_inference_steps=args.steps,
                generator=torch.Generator().manual_seed(args.seed + 1000 * t + k),
                label_format=args.format,
                image_name=f"{name}.jpg",
                output=["images", "line_cers", "labels_xml", "unclear_lines"],
            )
            state["images"][0].save(args.out / f"{name}.jpg", quality=92)
            (args.out / f"{name}.xml").write_text(state["labels_xml"], encoding="utf-8")
            crop = CropResult(
                image=f"{name}.jpg",
                lines=len(plan.lines),
                unclear_lines=state["unclear_lines"],
                cer=mean(state["line_cers"]),
            )
            crops.append(crop)
            print(f"{name}: {crop.lines} lines, CER {crop.cer:.3f}, {crop.unclear_lines} unclear", flush=True)
        pages.append(PageResult(template=str(template), crops=crops))

    summary = Summary(style=style, lora=args.lora, steps=args.steps, pages=pages)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
