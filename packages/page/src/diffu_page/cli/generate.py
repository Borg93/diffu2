"""Generate synthetic archive crops with ALTO/PAGE ground truth: any page template, any text, a chosen style.

Each template (a real PAGE/ALTO page) is planned on CPU (:func:`diffu_page.jobs.crop_jobs`): its running text is
replaced by the new text and it is cut into training-scale crops. Each crop is then drawn on its own by the crop
pipeline (FLUX.3 + LoRAs, or diffu), read back line by line, and labelled with badly read lines tagged ``unclear``,
as diffu-runner's GPU actors draw it. With ``--reference`` (a crop to draw in the hand of, for the v3 LoRAs) every
crop is drawn from it and captioned as the LoRA was trained (``--reference-caption``: ``described`` measures the
crop's hand, so it needs the crop's ``metadata.jsonl`` from ``diffu_page.data.build_crops`` beside it). Writes
``<out>/<page>_<k>.jpg`` + ``.xml`` per crop and ``<out>/summary.json``; a page that cannot be planned and a crop
that cannot be drawn are recorded there by name and the run goes on. ``--profile <dir>`` also records a torch
profiler trace (Perfetto) of crops 2-3 after a warm-up crop. One GPU job, through jobs/spark_guard.sh:

    jobs/spark_guard.sh --need-gb 45 --name pages -- diffu-generate \\
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
from diffusers.modular_pipelines import ModularPipeline
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, schedule

from diffu_page.blocks import DrawnCrop
from diffu_page.captions.look import ScanLook
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.cli.options import Lora, parse_lora, required_model
from diffu_page.jobs import CropJob, PageJob, crop_jobs
from diffu_page.layout import load_xml_layout, running_text
from diffu_page.output import console, err_console
from diffu_page.pipeline import DrawSettings, crop_blocks, draw_crop, load_pipeline
from diffu_page.plan import PageStyle, Reference
from diffu_page.settings import settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class CropResult(BaseModel):
    """A drawn crop: its name, its template and how well its lines read back."""

    name: str
    template: str
    lines: int
    unclear_lines: int
    cer: float


class Summary(BaseModel):
    """A run: what drew it (the model, its LoRAs, the seed, the draw settings, the style and the reference), every
    drawn crop, and every page or crop that failed, by name, with its error."""

    model: str
    loras: dict[str, str]
    seed: int
    settings: DrawSettings
    style: PageStyle
    reference: Reference | None
    crops: list[CropResult]
    failed: dict[str, str]


def _save(crop: DrawnCrop, template: Path, out: Path) -> CropResult:
    crop.image.save(out / f"{crop.name}.jpg", quality=92)
    (out / f"{crop.name}.xml").write_text(crop.labels_xml, encoding="utf-8")
    return CropResult(
        name=crop.name,
        template=str(template),
        lines=len(crop.plan.lines),
        unclear_lines=crop.unclear_lines,
        cer=fmean(crop.line_cers),
    )


def _draw(pipe: ModularPipeline, job: CropJob, *, settings: DrawSettings, out: Path) -> CropResult:
    crop = draw_crop(pipe, job.plan, name=job.name, seed=job.seed, reference=job.reference_image(), settings=settings)
    result = _save(crop, job.template, out)
    console.print(f"{crop.name}: CER {result.cer:.3f}, {crop.unclear_lines} unclear lines")
    return result


def _planned(pages: list[PageJob]) -> tuple[list[CropJob], dict[str, str]]:
    """Every page's crop jobs, and the error of each page that cannot be planned, by page name."""
    jobs: list[CropJob] = []
    failed: dict[str, str] = {}
    for index, page in enumerate(pages):
        try:
            jobs.extend(crop_jobs(page, index))
        except Exception as error:  # one page must not end a long run; the summary records it
            failed[page.name_at(index)] = f"{type(error).__name__}: {error}"
            err_console.print(f"{page.name_at(index)} failed: {failed[page.name_at(index)]}")
    return jobs, failed


def draw_pages(
    pipe: ModularPipeline, pages: list[PageJob], *, settings: DrawSettings, out: Path, prof: profile | None
) -> tuple[list[CropResult], dict[str, str]]:
    """Every crop of every page drawn by the crop pipeline and saved; a page that cannot be planned and a crop that
    cannot be drawn are recorded by name with their error, and the rest still run."""
    jobs, failed = _planned(pages)
    results: list[CropResult] = []
    for job in jobs:
        try:
            results.append(_draw(pipe, job, settings=settings, out=out))
        except Exception as error:  # one crop must not cost the page's other crops; the summary records it
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


def page_reference(image: Path | None, caption: ReferenceCaption) -> Reference | None:
    """The reference to draw every page from: for a described caption, a crop with its line boxes
    (``Reference.of_crop``)."""
    if image is None:
        return None
    return Reference.of_crop(image, caption) if caption is ReferenceCaption.DESCRIBED else Reference(image=image)


@app.command()
def generate(
    templates: Annotated[list[Path], typer.Option(help="a PAGE/ALTO page whose layout to use (repeat it)")],
    text_from: Annotated[list[Path], typer.Option(help="a PAGE/ALTO page whose text to write (repeat it)")],
    out: Annotated[Path, typer.Option(help="where the crops, labels and summary.json go")],
    title: Annotated[str, typer.Option(help="the archive, as the captions name it")],
    model: Annotated[
        Path | None, typer.Option(help="the model directory (FLUX.3 Klein or diffu); default $FLUX3_MODEL")
    ] = settings.flux3_model,
    lora: Annotated[list[Lora] | None, typer.Option(parser=parse_lora, help="name=<NeMo LoRA model dir>")] = None,
    years: str = "",
    greyscale: Annotated[bool, typer.Option("--greyscale/--colour")] = False,
    paper: str = "cream",
    ink: str = "dark brown",
    reference: Annotated[
        Path | None, typer.Option(help="a crop to draw every page in the hand, ink and paper of (the v3 LoRAs)")
    ] = None,
    reference_caption: Annotated[
        ReferenceCaption,
        typer.Option(help="the caption form the LoRA was trained on: plain (v3, v3c), described (v3d)"),
    ] = ReferenceCaption.PLAIN,
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
    model = required_model(model)
    style = PageStyle(
        provenance=Provenance(title=title, years=years), look=ScanLook(greyscale=greyscale, paper=paper, ink=ink)
    )
    text = " ".join(running_text(load_xml_layout(path)) for path in text_from)
    drawn_from = page_reference(reference, reference_caption)
    pages = [
        PageJob(template=template, text=text, style=style, seed=seed, reference=drawn_from) for template in templates
    ]
    draw_settings = DrawSettings(
        steps=steps,
        guidance_scale=guidance,
        label_format=label_format,
        text_layers=tuple(text_layer) if text_layer else None,
        device=device,
    )
    loras = {adapter.name: adapter.path for adapter in lora or []}
    pipe = load_pipeline(model, crop_blocks, loras=loras, settings=draw_settings)
    out.mkdir(parents=True, exist_ok=True)
    with _profiler(profile_dir) as prof:
        crops, failed = draw_pages(pipe, pages, settings=draw_settings, out=out, prof=prof)
    summary = Summary(
        model=str(model),
        loras={name: str(path) for name, path in loras.items()},
        seed=seed,
        settings=draw_settings,
        style=style,
        reference=drawn_from,
        crops=crops,
        failed=failed,
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")
    console.print(f"{len(crops)} crops drawn, {len(failed)} pages or crops failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
