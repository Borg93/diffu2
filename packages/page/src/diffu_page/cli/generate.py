"""Generate synthetic archive crops or whole pages with ALTO/PAGE ground truth: any page template, any text, a chosen
style.

Each template (a real PAGE/ALTO page, or one written for the purpose) is planned on CPU
(:func:`diffu_page.jobs.crop_jobs`): its running text is replaced by the new text and it is cut into training-scale
crops, or with ``--whole-page`` kept whole as one crop of ``--page-megapixels``. With ``--mark-from-scan`` a
ground-truth template is first marked from the scan beside it (:func:`diffu_page.marking.marked`): the lines its PAGE
file does not say are printed or handwritten read off the scan by ``--print-hand-model``, and its ruled lines found
there if it has none, so its form's print is drawn as print and its grid as rules. A two-page opening (most scans) is
planned as its single pages, never as a spread (:func:`diffu_page.jobs.single_pages`), and ``summary.json`` lists them.
Each crop is then drawn on its own by the crop pipeline (FLUX.3 + LoRAs, or diffu), read back line by line, and labelled
with badly read lines tagged ``unclear``, as diffu-runner's GPU actors draw it. The text comes from ``--text-from``: a
PAGE/ALTO page's running text, or a plain text file as it stands (a blank line ends a paragraph). With ``--reference``
(a crop to draw in the hand of, for the v3 LoRAs) every crop is drawn from it and captioned as the LoRA was trained
(``--reference-caption``: ``described`` measures the crop's hand, so it needs the crop's ``metadata.jsonl`` from
``diffu_page.data.build_crops`` beside it). ``--jobs`` gives each page its own template, text, style, seed, reference,
size and side instead: one :class:`diffu_page.jobs.PageJob` per line, the run file diffu-runner reads. A style can be
named instead of spelled out: ``--style`` (or a job line's ``style``) names a preset of ``--styles``
(:mod:`diffu_page.styles`: the archive and years, the look, the hand's LoRA and a reference), and its page is drawn by
the preset's hand alone; a style spelled out is drawn with ``--lora``. A run can so draw several hands: its pages are
drawn group by group, one group per set of LoRAs, each group's pipeline loaded once and released before the next loads.
Writes ``<out>/<page>_<k>.jpg`` + ``.xml`` per crop and ``<out>/summary.json``; a page that cannot be planned and a crop
that cannot be drawn are recorded there by name and the run goes on. ``--profile <dir>`` also records a torch profiler
trace (Perfetto) of crops 2-3 after a warm-up crop. One GPU job, through jobs/spark_guard.sh:

    jobs/spark_guard.sh --need-gb 45 --name pages -- diffu-generate \\
        --templates <page.xml> --templates <page.xml> --text-from <page.xml> --lora v2=<ckpt>/model --out <dir> \\
        --title "the court records of Jämtlands domsaga" --paper cream --ink "dark brown"
    jobs/spark_guard.sh --need-gb 45 --name pages -- diffu-generate \\
        --templates <page.xml> --text-from <text.txt> --styles styles.json --style pahlman --out <dir>
    jobs/spark_guard.sh --need-gb 45 --name pages -- diffu-generate --jobs run.jsonl --styles styles.json --out <dir>
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from statistics import fmean
from typing import Annotated

import torch
import typer
from click.core import ParameterSource
from diffusers.modular_pipelines import ModularPipeline
from pydantic import BaseModel
from torch.profiler import ProfilerActivity, profile, schedule

from diffu_page.blocks import DrawnCrop
from diffu_page.captions.look import Ink, Paper, ScanLook
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import Provenance
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.cli.options import (
    DEFAULTS,
    CompileBlocks,
    Device,
    Format,
    FuseLoras,
    Guidance,
    GuidanceStop,
    Loras,
    LoraScale,
    MaxLineCer,
    Model,
    ReferenceGuidanceScale,
    RepairPasses,
    SamplingShift,
    Steps,
    TextLayers,
    lora_dirs,
    required_model,
    settings_from_options,
)
from diffu_page.jobs import CropJob, PageJob, PageRequest, ResolvedPage, crop_jobs, read_page_jobs, single_pages
from diffu_page.judge import Judge, TrOCRJudge
from diffu_page.layout import load_xml_layout, running_text
from diffu_page.local_run import run_each
from diffu_page.marking import PrintHandClassifier
from diffu_page.output import JPEG_QUALITY, console
from diffu_page.pipeline import (
    DEVICE_ERRORS,
    DrawSettings,
    crop_blocks,
    draw_crop,
    load_pipeline,
    repair_lines,
    repair_seed,
)
from diffu_page.plan import Reference
from diffu_page.settings import settings
from diffu_page.styles import StylePreset, read_style_presets

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

PAGE_OPTIONS = (  # what a job file gives each page instead
    "templates",
    "text_from",
    "title",
    "style",
    "years",
    "greyscale",
    "paper",
    "ink",
    "reference",
    "reference_caption",
    "seed",
    "whole_page",
    "page_megapixels",
)
STYLE_OPTIONS = ("title", "years", "greyscale", "paper", "ink")  # what a style preset gives instead (--style)

type LoraJob = tuple[dict[str, Path], CropJob]  # a crop job and the LoRAs that draw it, by adapter name


class CropSummary(BaseModel):
    """A drawn crop as ``summary.json`` lists it: its name, its template, how well its lines read back, and how many
    of them its repair passes took from unclear to clear (net)."""

    name: str
    template: str
    lines: int
    unclear_lines: int
    cer: float
    repaired: int = 0


class PageGroup(BaseModel):
    """Pages a run draws with one crop pipeline: the LoRAs it loads (NeMo ``model/`` dirs by adapter name) and the
    pages, by their place in the run (in ``Summary.pages``)."""

    loras: dict[str, Path]
    pages: list[int]


class Summary(BaseModel):
    """A run: what drew it (the model, the LoRAs of each group of its pages and the draw settings), the print/hand
    classifier its templates were marked from their scans with (None: they were not), the pages it was asked for (each
    with its template, text, style, seed and reference, a named style resolved), every drawn crop, and every page or
    crop that failed, by name, with its error."""

    model: str
    groups: list[PageGroup]
    settings: DrawSettings
    marked_with: Path | None = None
    pages: list[PageJob]
    crops: list[CropSummary]
    failed: dict[str, str]


def _save(crop: DrawnCrop, template: Path, out: Path, *, repaired: int) -> CropSummary:
    crop.image.save(out / f"{crop.name}.jpg", quality=JPEG_QUALITY)
    (out / f"{crop.name}.xml").write_text(crop.labels_xml, encoding="utf-8")
    return CropSummary(
        name=crop.name,
        template=str(template),
        lines=len(crop.plan.lines),
        unclear_lines=crop.unclear_lines,
        cer=fmean(crop.line_cers),
        repaired=repaired,
    )


def _draw(
    pipe: ModularPipeline, job: CropJob, *, settings: DrawSettings, out: Path, judge: Judge | None
) -> CropSummary:
    reference = job.reference_image()
    crop = draw_crop(pipe, job.plan, name=job.name, seed=job.seed, reference=reference, settings=settings, judge=judge)
    drawn_unclear = crop.unclear_lines
    for repair_pass in range(1, settings.repair_passes + 1):
        crop = repair_lines(
            pipe, crop, seed=repair_seed(job.seed, repair_pass), reference=reference, settings=settings, judge=judge
        )
    result = _save(crop, job.template, out, repaired=drawn_unclear - crop.unclear_lines)
    console.print(f"{crop.name}: CER {result.cer:.3f}, {crop.unclear_lines} unclear lines ({result.repaired} repaired)")
    return result


def page_groups(pages: list[ResolvedPage]) -> list[PageGroup]:
    """The run's pages grouped by the LoRAs that draw them, in the order each set of LoRAs first appears."""
    groups: dict[frozenset[tuple[str, Path]], PageGroup] = {}
    for index, page in enumerate(pages):
        groups.setdefault(frozenset(page.loras.items()), PageGroup(loras=page.loras, pages=[])).pages.append(index)
    return list(groups.values())


def plan_jobs(
    pages: Iterable[tuple[int, PageJob]], *, print_hand: PrintHandClassifier | None = None
) -> tuple[list[CropJob], dict[str, str]]:
    """The crop jobs of each page, given with its place in the run (which names and seeds its crops), its template
    marked from its scan with ``print_hand`` if one is given, and the error of each page that cannot be planned, by page
    name (CPU)."""
    planned, failed = run_each(
        list(pages),
        lambda indexed: crop_jobs(indexed[1], indexed[0], print_hand=print_hand),
        key=lambda indexed: indexed[1].name_at(indexed[0]),
    )
    return [job for jobs in planned for job in jobs], failed


def draw_pages(
    load: Callable[[dict[str, Path]], ModularPipeline],
    jobs: list[LoraJob],
    *,
    settings: DrawSettings,
    out: Path,
    prof: profile | None,
    judge: Judge | None = None,
) -> tuple[list[CropSummary], dict[str, str]]:
    """Every planned crop drawn, in the order given, by the crop pipeline ``load`` gives for its LoRAs, judged by
    ``judge`` (the default TrOCR judge if none) and saved. One pipeline is held at a time: loaded for the first crop
    and again for each crop whose LoRAs differ from the crop's before, the previous one released first, so give the
    crops grouped by their LoRAs. A crop that cannot be drawn, or whose pipeline cannot be loaded, is recorded by name
    with its error and the rest still run, unless the device failed: then every crop after it is recorded as not run
    (:func:`diffu_page.local_run.run_each`)."""
    pipe: ModularPipeline | None = None
    pipe_loras: dict[str, Path] = {}

    def draw(item: LoraJob) -> CropSummary:
        nonlocal pipe, pipe_loras
        loras, job = item
        if pipe is None or loras != pipe_loras:
            pipe = None  # released before the next one loads: two pipelines never share the device
            _release_device_memory(settings.device)
            pipe, pipe_loras = load(loras), loras
        result = _draw(pipe, job, settings=settings, out=out, judge=judge)
        if prof is not None:
            prof.step()
        return result

    return run_each(jobs, draw, key=lambda item: item[1].name, stop_on=DEVICE_ERRORS)


def _release_device_memory(device: str) -> None:
    """The memory of the models dropped so far given back: collected now rather than at Python's next collection, and
    CUDA's cached blocks returned on a CUDA device."""
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


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


def print_hand_classifier(weights: Path) -> PrintHandClassifier:
    """The print/hand classifier saved at ``weights``.

    Raises:
        typer.BadParameter: If no file is there.
    """
    if not weights.is_file():
        raise typer.BadParameter(
            f"no print/hand classifier at {weights}: train one with python -m diffu_page.marking.train",
            param_hint="--print-hand-model",
        )
    return PrintHandClassifier.load(weights)


def page_reference(image: Path | None, caption: ReferenceCaption) -> Reference | None:
    """The reference to draw every page from: for a described caption, a crop with its line boxes
    (``Reference.of_crop``)."""
    if image is None:
        return None
    return Reference.of_crop(image, caption) if caption is ReferenceCaption.DESCRIBED else Reference(image=image)


def page_text(path: Path) -> str:
    """The text to write from ``path``: a PAGE/ALTO page's running text, or a text file's text as it stands (a blank
    line ends a paragraph)."""
    if path.suffix.lower() == ".xml":
        return running_text(load_xml_layout(path))
    return path.read_text(encoding="utf-8").strip()


def refuse_given(ctx: typer.Context, names: Sequence[str], *, instead: str, param_hint: str) -> None:
    """Refuse each option of ``names`` (by parameter name) the command line gives; ``instead`` says what gives it.

    Raises:
        typer.BadParameter: If any of them is given, naming them.
    """
    given = [name for name in names if ctx.get_parameter_source(name) is not ParameterSource.DEFAULT]
    if given:
        options = ", ".join("--" + name.replace("_", "-") for name in given)
        raise typer.BadParameter(f"{options}: {instead}", param_hint=param_hint)


def style_presets(styles: Path | None) -> dict[str, StylePreset]:
    """The style presets of ``--styles``; none without it."""
    if styles is None:
        return {}
    try:
        return read_style_presets(styles)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--styles") from error


def flag_style(
    ctx: typer.Context, *, style: str | None, title: str | None, years: str, greyscale: bool, paper: Paper, ink: Ink
) -> PageStyle | str | None:
    """The style the options ask for: a style preset's name (``--style``, refusing the options a preset gives), else
    the style ``--title`` and the look options spell out; None if they name neither."""
    if style is not None:
        refuse_given(ctx, STYLE_OPTIONS, instead="the style preset gives its own", param_hint="--style")
        return style
    if title is None:
        return None
    return PageStyle(
        provenance=Provenance(title=title, years=years), look=ScanLook(greyscale=greyscale, paper=paper, ink=ink)
    )


def job_file_pages(
    ctx: typer.Context, jobs: Path, *, presets: dict[str, StylePreset], loras: dict[str, Path]
) -> list[ResolvedPage]:
    """The pages of the job file ``jobs``, resolved with the style ``presets`` and the run's ``loras``, refusing every
    page option the command line gives with it: the job file gives each page its own."""
    refuse_given(ctx, PAGE_OPTIONS, instead="the job file gives each page its own", param_hint="--jobs")
    try:
        pages = read_page_jobs(jobs, presets=presets, loras=loras)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--jobs") from error
    if not pages:
        raise typer.BadParameter(f"{jobs} holds no page jobs", param_hint="--jobs")
    return pages


def plan_groups(
    pages: list[ResolvedPage], groups: list[PageGroup], *, print_hand: PrintHandClassifier | None = None
) -> tuple[list[LoraJob], dict[str, str]]:
    """Each group's pages planned into crop jobs with the group's LoRAs, group after group (each template marked from
    its scan with ``print_hand`` if one is given), and the error of each page that cannot be planned, by page name
    (CPU)."""
    jobs: list[LoraJob] = []
    failed: dict[str, str] = {}
    for group in groups:
        planned, unplanned = plan_jobs(((index, pages[index].page) for index in group.pages), print_hand=print_hand)
        jobs += [(group.loras, job) for job in planned]
        failed |= unplanned
    return jobs, failed


@app.command()
def generate(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option(help="where the crops, labels and summary.json go")],
    job_file: Annotated[
        Path | None,
        typer.Option(
            "--jobs",
            exists=True,
            dir_okay=False,
            help="the pages to draw, one diffu_page.jobs.PageJob per line (diffu-run's run file); a line's style may "
            "name a preset of --styles",
        ),
    ] = None,
    styles: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="the style presets --style and a job file's named styles come from: a JSON object of "
            "diffu_page.styles.StylePreset by name, such as the repo's styles.json",
        ),
    ] = None,
    templates: Annotated[
        list[Path] | None, typer.Option(help="a PAGE/ALTO page whose layout to use (repeat it)")
    ] = None,
    text_from: Annotated[
        list[Path] | None,
        typer.Option(help="a PAGE/ALTO page (.xml) or a plain text file whose text to write (repeat it)"),
    ] = None,
    title: Annotated[str | None, typer.Option(help="the archive, as the captions name it")] = None,
    style: Annotated[
        str | None,
        typer.Option(
            help="a preset of --styles to draw every page in, its hand and reference with it, instead of --title and "
            "the look options (--reference still gives the pages their own)"
        ),
    ] = None,
    model: Model = settings.flux3_model,
    lora: Loras = None,
    years: str = "",
    greyscale: Annotated[bool, typer.Option("--greyscale/--colour")] = False,
    paper: Annotated[Paper, typer.Option(help="the paper's colour, as the captions name it")] = Paper.CREAM,
    ink: Annotated[Ink, typer.Option(help="the ink's colour, as the captions name it")] = Ink.DARK_BROWN,
    reference: Annotated[
        Path | None, typer.Option(help="a crop to draw every page in the hand, ink and paper of (the v3 LoRAs)")
    ] = None,
    reference_caption: Annotated[
        ReferenceCaption,
        typer.Option(help="the caption form the LoRA was trained on: plain (v3, v3c), described (v3d)"),
    ] = ReferenceCaption.PLAIN,
    steps: Steps = DEFAULTS.steps,
    guidance: Guidance = DEFAULTS.guidance_scale,
    reference_guidance: ReferenceGuidanceScale = DEFAULTS.reference_guidance_scale,
    guidance_stop: GuidanceStop = DEFAULTS.guidance_stop,
    seed: int = 0,
    text_layer: TextLayers = None,
    device: Device = DEFAULTS.device,
    max_line_cer: MaxLineCer = DEFAULTS.max_line_cer,
    label_format: Format = DEFAULTS.label_format,
    lora_scale: LoraScale = DEFAULTS.lora_scale,
    sampling_shift: SamplingShift = DEFAULTS.sampling_shift,
    fuse_loras: FuseLoras = DEFAULTS.fuse_loras,
    compile_blocks: CompileBlocks = DEFAULTS.compile_blocks,
    repair: RepairPasses = DEFAULTS.repair_passes,
    whole_page: Annotated[
        bool, typer.Option("--whole-page/--windows", help="draw each page whole instead of as training-scale windows")
    ] = False,
    page_megapixels: Annotated[
        float, typer.Option(min=0.25, max=4.0, help="a whole page's size (FLUX.3 draws up to about 4 MP)")
    ] = 2.5,
    mark_from_scan: Annotated[
        bool,
        typer.Option(
            "--mark-from-scan",
            help="mark each template from the scan beside it before planning: the lines its PAGE file does not say "
            "are printed or handwritten read off the scan, and its ruled lines found there if it has none",
        ),
    ] = False,
    print_hand_model: Annotated[
        Path, typer.Option(help="the print/hand classifier --mark-from-scan reads lines with")
    ] = settings.print_hand_model,
    profile_dir: Annotated[Path | None, typer.Option("--profile", help="write a trace of crops 2-3 here")] = None,
) -> None:
    """Draw every crop of the templates (or each template whole), refilled with the text, in the given or named style;
    or every page of a job file, each in its own."""
    model_dir = required_model(model)
    draw_settings = settings_from_options(
        steps=steps,
        guidance=guidance,
        reference_guidance=reference_guidance,
        guidance_stop=guidance_stop,
        text_layer=text_layer,
        device=device,
        max_line_cer=max_line_cer,
        lora_scale=lora_scale,
        sampling_shift=sampling_shift,
        fuse_loras=fuse_loras,
        compile_blocks=compile_blocks,
        label_format=label_format,
        repair_passes=repair,
    )
    presets = style_presets(styles)
    loras = lora_dirs(lora)
    print_hand = print_hand_classifier(print_hand_model) if mark_from_scan else None
    if job_file is not None:
        pages = job_file_pages(ctx, job_file, presets=presets, loras=loras)
    else:
        page_style = flag_style(ctx, style=style, title=title, years=years, greyscale=greyscale, paper=paper, ink=ink)
        if not templates or not text_from or page_style is None:
            raise typer.BadParameter(
                "give --jobs, or --templates, --text-from and --title or --style", param_hint="the pages"
            )
        if reference is None and reference_caption is not ReferenceCaption.PLAIN:
            raise typer.BadParameter("there is no --reference to describe", param_hint="--reference-caption")
        text = " ".join(page_text(path) for path in text_from)
        drawn_from = page_reference(reference, reference_caption)
        page_pixels = round(page_megapixels * 1_000_000) if whole_page else None
        requests = [
            PageRequest(
                template=template, text=text, style=page_style, seed=seed, reference=drawn_from, page_pixels=page_pixels
            )
            for template in templates
        ]
        try:
            pages = [request.resolve(presets, loras) for request in requests]
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint="--style") from error
    if loras and all(page.loras != loras for page in pages):
        raise typer.BadParameter(
            "every page names a style preset, which gives the LoRAs it is drawn with", param_hint="--lora"
        )
    if draw_settings.guides_reference and all(page.page.reference is None for page in pages):
        raise typer.BadParameter("there is no --reference to guide toward", param_hint="--reference-guidance")
    pages = [ResolvedPage(page=single, loras=page.loras) for page in pages for single in single_pages(page.page)]
    groups = page_groups(pages)
    # before any model loads: a run whose pages plan to nothing loads none
    jobs, failed = plan_groups(pages, groups, print_hand=print_hand)
    crops: list[CropSummary] = []
    out.mkdir(parents=True, exist_ok=True)
    if jobs:
        judge = TrOCRJudge(device=device)
        with _profiler(profile_dir) as prof:
            crops, draw_failed = draw_pages(
                lambda group_loras: load_pipeline(model_dir, crop_blocks, loras=group_loras, settings=draw_settings),
                jobs,
                settings=draw_settings,
                out=out,
                prof=prof,
                judge=judge,
            )
        failed |= draw_failed
    summary = Summary(
        model=str(model_dir),
        groups=groups,
        settings=draw_settings,
        marked_with=print_hand_model if mark_from_scan else None,
        pages=[page.page for page in pages],
        crops=crops,
        failed=failed,
    )
    (out / "summary.json").write_text(summary.model_dump_json(indent=1), encoding="utf-8")
    console.print(f"{len(crops)} crops drawn, {len(failed)} pages or crops failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
