"""Generate synthetic archive crops with ALTO/PAGE ground truth: any page template, any text, a chosen style.

For each template (a real PAGE/ALTO page) the running text is replaced by new text (``refill``), the page is planned
as training-scale crops (``page_synth.plan``), and every crop goes through a :class:`page_synth.generator.CropGenerator`:
FLUX.3 + LoRAs draw it, TrOCR reads every line back, and its labels are written with badly read lines tagged
``unclear``. Writes ``<out>/<page>_<k>.jpg`` + ``.xml`` per crop and ``<out>/summary.json``; ``--profile <dir>`` also
records a torch profiler trace (Perfetto) of crops 2-3 after a warm-up crop. One GPU job, through
jobs/spark_guard.sh:

    jobs/spark_guard.sh --need-gb 45 --name pages -- python -m page_synth.cli.generate \\
        --templates <page.xml ...> --text-from <page.xml ...> --lora v2=<ckpt>/model --out <dir> \\
        --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from page_synth.captions.look import ScanLook
from page_synth.captions.provenance import Provenance
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.layout import load_xml_layout, refill, running_text
from page_synth.modular.pipelines import FLUX3_KLEIN
from page_synth.modular.readback import mean
from page_synth.plan import PageStyle, plan_page


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
    loras: dict[str, str]
    steps: int
    pages: list[PageResult]


def lora_arg(value: str) -> tuple[str, Path]:
    """``name=<dir>`` as an adapter name and a NeMo LoRA checkpoint ``model/`` dir."""
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected name=<lora model dir>, got {value!r}")
    return name, Path(path)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", type=Path, nargs="+", required=True, help="PAGE/ALTO pages whose layout to use")
    ap.add_argument("--text-from", type=Path, nargs="+", required=True, help="PAGE/ALTO pages whose text to write")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=FLUX3_KLEIN)
    ap.add_argument("--lora", type=lora_arg, action="append", default=[], help="name=<NeMo LoRA model dir>")
    ap.add_argument("--title", required=True, help="the archive, as the captions name it")
    ap.add_argument("--years", default="")
    ap.add_argument("--greyscale", action="store_true")
    ap.add_argument("--paper", default="cream")
    ap.add_argument("--ink", default="dark brown")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--format", choices=["alto", "page"], default="alto")
    ap.add_argument("--profile", type=Path, default=None, help="write a profiler trace of crops 2-3 here")
    return ap.parse_args()


def _trace_handler(out: Path) -> Callable[[profile], None]:
    def handle(prof: profile) -> None:
        out.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(out / "trace.json"))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        (out / "top_ops.txt").write_text(table, encoding="utf-8")
        print(table, flush=True)

    return handle


def _profiler(out: Path | None) -> AbstractContextManager[profile | None]:
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
    text = " ".join(running_text(load_xml_layout(path)) for path in args.text_from)
    args.out.mkdir(parents=True, exist_ok=True)
    loras = dict(args.lora)
    settings = GeneratorSettings(steps=args.steps, guidance_scale=args.guidance, label_format=args.format)
    generate = CropGenerator(args.model, loras=loras, settings=settings)

    pages: list[PageResult] = []
    with _profiler(args.profile) as prof:
        for t, template in enumerate(args.templates):
            layout = refill(load_xml_layout(template), [text])
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

    summary = Summary(
        style=style, loras={name: str(path) for name, path in loras.items()}, steps=args.steps, pages=pages
    )
    (args.out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
