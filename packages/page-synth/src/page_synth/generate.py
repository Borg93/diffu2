"""Generate synthetic archive crops with ALTO/PAGE ground truth: any page template, any text, a chosen style.

For each template (a real PAGE/ALTO page, replayed with diffu-page) the running text is replaced by new text
(``refill``), the page is planned as training-scale crops (``page_synth.plan``), and every crop goes through a
:class:`CropGenerator`: FLUX.3 Klein + a LoRA draws it, TrOCR reads every line back, and its labels are written with
badly-read lines tagged ``unclear``. ``CropGenerator`` is the unit a Ray Data actor wraps: the pipeline is loaded
once, then each call turns one crop plan into one image + labels. Writes ``<out>/<page>_<k>.jpg`` + ``.xml`` per crop
and ``<out>/summary.json``; ``--profile <dir>`` also records a torch profiler trace (Perfetto) of crops 2-3 after a
warm-up crop. Runs in the NeMo AutoModel environment, one GPU job, through scripts/spark_guard.sh:

    cd ~/Desktop/repos/Automodel && ~/Desktop/repos/diffu2/scripts/spark_guard.sh --need-gb 45 --name pages -- \\
        python -m page_synth.generate \\
        --templates <page.xml ...> --text-from <page.xml ...> --lora <ckpt>/model --out <dir> \\
        --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from diffu_page import load_xml_layout, refill, running_text
from PIL import Image
from pydantic import BaseModel, ConfigDict
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from page_synth.blocks import DEFAULT_MODEL, LabelFormat, apply_nemo_lora, build_generator, mean
from page_synth.look import ScanLook
from page_synth.plan import CropPlan, PageStyle, plan_page
from page_synth.provenance import Provenance


class GeneratedCrop(BaseModel):
    """One drawn crop: its image, its labels and how well its lines read back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Image.Image
    labels_xml: str
    line_cers: list[float]
    unclear_lines: int
    seconds: float


class CropGenerator:
    """FLUX.3 Klein (+ LoRA) as a crop generator: loaded once, then one crop plan in, one crop out per call."""

    def __init__(self, model: str, *, lora: str | None, steps: int, label_format: LabelFormat) -> None:
        self._pipe = build_generator(model)
        if lora:
            print(
                f"LoRA: {apply_nemo_lora(self._pipe.transformer, Path(lora))} modules from {lora}", flush=True
            )
        self._steps = steps
        self._label_format = label_format

    def __call__(self, plan: CropPlan, *, seed: int, image_name: str) -> GeneratedCrop:
        start = time.perf_counter()
        state = self._pipe(
            crop_plan=plan,
            num_inference_steps=self._steps,
            generator=torch.Generator().manual_seed(seed),
            label_format=self._label_format,
            image_name=image_name,
            output=["images", "line_cers", "labels_xml", "unclear_lines"],
        )
        return GeneratedCrop(
            image=state["images"][0],
            labels_xml=state["labels_xml"],
            line_cers=state["line_cers"],
            unclear_lines=state["unclear_lines"],
            seconds=time.perf_counter() - start,
        )


class CropResult(BaseModel):
    image: str
    lines: int
    unclear_lines: int
    cer: float
    seconds: float


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
    ap.add_argument("--profile", type=Path, default=None, help="write a profiler trace of crops 2-3 here")
    return ap.parse_args()


def _trace_handler(out: Path):
    def handle(prof: profile) -> None:
        out.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(out / "trace.json"))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        (out / "top_ops.txt").write_text(table, encoding="utf-8")
        print(table, flush=True)

    return handle


def _profiler(out: Path | None):
    """One warm-up crop, then two recorded ones (CPU + CUDA, shapes, memory), or nothing."""
    if out is None:
        return nullcontext(None)
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=2, repeat=1),
        on_trace_ready=_trace_handler(out),
        record_shapes=True,
        profile_memory=True,
    )


def main() -> None:
    args = _parse_args()
    style = PageStyle(
        provenance=Provenance(title=args.title, years=args.years),
        look=ScanLook(greyscale=args.greyscale, paper=args.paper, ink=args.ink),
    )
    text = " ".join(running_text(load_xml_layout(str(path), single_page=False)) for path in args.text_from)
    args.out.mkdir(parents=True, exist_ok=True)
    generate = CropGenerator(args.model, lora=args.lora, steps=args.steps, label_format=args.format)

    pages: list[PageResult] = []
    with _profiler(args.profile) as prof:
        for t, template in enumerate(args.templates):
            layout = refill(load_xml_layout(str(template), single_page=False), [text])
            crops: list[CropResult] = []
            for k, plan in enumerate(plan_page(layout, style)):
                name = f"{t:03d}_{template.stem}_{k}"
                with record_function(f"crop {plan.width}x{plan.height}"):
                    crop = generate(plan, seed=args.seed + 1000 * t + k, image_name=f"{name}.jpg")
                if prof is not None:
                    prof.step()
                crop.image.save(args.out / f"{name}.jpg", quality=92)
                (args.out / f"{name}.xml").write_text(crop.labels_xml, encoding="utf-8")
                result = CropResult(
                    image=f"{name}.jpg",
                    lines=len(plan.lines),
                    unclear_lines=crop.unclear_lines,
                    cer=mean(crop.line_cers),
                    seconds=crop.seconds,
                )
                crops.append(result)
                print(
                    f"{name}: {result.lines} lines, CER {result.cer:.3f}, {result.unclear_lines} unclear, "
                    f"{result.seconds:.1f} s",
                    flush=True,
                )
            pages.append(PageResult(template=str(template), crops=crops))

    summary = Summary(style=style, lora=args.lora, steps=args.steps, pages=pages)
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
