"""Generate synthetic archive crops with ALTO/PAGE ground truth: any page template, any text, a chosen style.

For each template (a real PAGE/ALTO page) the running text is replaced by new text (``refill``), the page is planned
as training-scale crops (``page_synth.plan``), and every crop goes through a :class:`page_synth.generator.CropGenerator`:
FLUX.3 + LoRAs draw it, TrOCR reads every line back, and its labels are written with badly read lines tagged
``unclear``. Writes ``<out>/<page>_<k>.jpg`` + ``.xml`` per crop and ``<out>/summary.json``; a crop that fails is
recorded there and the run goes on. ``--profile <dir>`` also records a torch profiler trace (Perfetto) of crops 2-3
after a warm-up crop. One GPU job, through jobs/spark_guard.sh:

    jobs/spark_guard.sh --need-gb 45 --name pages -- page-synth-generate \\
        --templates <page.xml> --templates <page.xml> --text-from <page.xml> --lora v2=<ckpt>/model --out <dir> \\
        --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from statistics import fmean
from typing import Annotated, Literal

import typer
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from page_synth.captions.look import ScanLook
from page_synth.captions.provenance import Provenance
from page_synth.cli.options import Lora, parse_lora
from page_synth.cli.output import console, err_console
from page_synth.generator import CropGenerator, GeneratorSettings
from page_synth.layout import load_xml_layout, refill, running_text
from page_synth.plan import CropPlan, PageStyle, plan_page
from page_synth.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class CropJob(BaseModel):
    """One crop to draw: its name, the template it comes from, its plan and its noise seed."""

    name: str
    template: Path
    plan: CropPlan
    seed: int


class CropResult(BaseModel):
    """A drawn crop: its name, its template and how well its lines read back."""

    name: str
    template: str
    lines: int
    unclear_lines: int
    cer: float
    seconds: float


class Summary(BaseModel):
    """A run: its style, LoRAs and steps, every drawn crop, and every crop that failed with its error."""

    style: PageStyle
    loras: dict[str, str]
    steps: int
    crops: list[CropResult]
    failed: dict[str, str]


def plan_jobs(templates: list[Path], text: str, style: PageStyle, seed: int) -> list[CropJob]:
    """Every crop of every template refilled with ``text``; template ``t``'s crop ``k`` draws from noise seed
    ``seed + 1000 t + k``."""
    return [
        CropJob(name=f"{t:03d}_{template.stem}_{k}", template=template, plan=plan, seed=seed + 1000 * t + k)
        for t, template in enumerate(templates)
        for k, plan in enumerate(plan_page(refill(load_xml_layout(template), [text]), style))
    ]


def _draw(generator: CropGenerator, job: CropJob, out: Path) -> CropResult:
    with record_function(f"crop {job.plan.width}x{job.plan.height}"):
        crop = generator(job.plan, seed=job.seed, image_name=f"{job.name}.jpg")
    crop.image.save(out / f"{job.name}.jpg", quality=92)
    (out / f"{job.name}.xml").write_text(crop.labels_xml, encoding="utf-8")
    return CropResult(
        name=job.name,
        template=str(job.template),
        lines=len(job.plan.lines),
        unclear_lines=crop.unclear_lines,
        cer=fmean(crop.line_cers),
        seconds=crop.seconds,
    )


def draw_all(
    generator: CropGenerator, jobs: list[CropJob], out: Path, prof: profile | None
) -> tuple[list[CropResult], dict[str, str]]:
    """Every job drawn and saved; a crop that fails is recorded with its error and the rest still run."""
    results: list[CropResult] = []
    failed: dict[str, str] = {}
    for job in jobs:
        try:
            results.append(_draw(generator, job, out))
            console.print(f"{job.name}: CER {results[-1].cer:.3f}, {results[-1].unclear_lines} unclear lines")
        except Exception as error:  # one crop must not end a long run; the summary records it
            failed[job.name] = f"{type(error).__name__}: {error}"
            err_console.print(f"{job.name} failed: {failed[job.name]}")
        if prof is not None:
            prof.step()
    return results, failed


def _trace_handler(out: Path) -> Callable[[profile], None]:
    def handle(prof: profile) -> None:
        out.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(out / "trace.json"))
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
        (out / "top_ops.txt").write_text(table, encoding="utf-8")
        console.print(table)

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


@app.command()
def generate(
    templates: Annotated[list[Path], typer.Option(help="a PAGE/ALTO page whose layout to use (repeat it)")],
    text_from: Annotated[list[Path], typer.Option(help="a PAGE/ALTO page whose text to write (repeat it)")],
    out: Annotated[Path, typer.Option(help="where the crops, labels and summary.json go")],
    title: Annotated[str, typer.Option(help="the archive, as the captions name it")],
    model: Annotated[Path, typer.Option(help="the FLUX.3 model directory")] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    years: str = "",
    greyscale: Annotated[bool, typer.Option("--greyscale/--colour")] = False,
    paper: str = "cream",
    ink: str = "dark brown",
    steps: int = 50,
    guidance: float = 4.0,
    seed: int = 0,
    text_layer: Annotated[
        list[int] | None, typer.Option(help="a text-encoder layer to stack (repeat it; default: the model's own)")
    ] = None,
    device: Annotated[str, typer.Option(help="where the models run")] = "cuda",
    label_format: Annotated[
        Literal["alto", "page"], typer.Option("--format")
    ] = "alto",  # Typer cannot read LabelFormat
    profile_dir: Annotated[Path | None, typer.Option("--profile", help="write a trace of crops 2-3 here")] = None,
) -> None:
    """Draw every crop of the templates, refilled with the text, in the given style."""
    style = PageStyle(
        provenance=Provenance(title=title, years=years), look=ScanLook(greyscale=greyscale, paper=paper, ink=ink)
    )
    text = " ".join(running_text(load_xml_layout(path)) for path in text_from)
    jobs = plan_jobs(templates, text, style, seed)
    out.mkdir(parents=True, exist_ok=True)
    loras = {adapter.name: adapter.path for adapter in lora or []}
    generator_settings = GeneratorSettings(
        steps=steps,
        guidance_scale=guidance,
        label_format=label_format,
        text_layers=tuple(text_layer) if text_layer else None,
        device=device,
    )
    generator = CropGenerator(model, loras=loras, settings=generator_settings)
    with _profiler(profile_dir) as prof:
        crops, failed = draw_all(generator, jobs, out, prof)
    summary = Summary(
        style=style, loras={name: str(path) for name, path in loras.items()}, steps=steps, crops=crops, failed=failed
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")
    console.print(f"{len(crops)} crops drawn, {len(failed)} failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
