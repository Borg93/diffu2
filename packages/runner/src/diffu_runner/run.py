"""A generation run across a Ray cluster: a run file's pages drawn whole by GPU actors, each holding one pipeline, and
committed to the run's Lance table as they are drawn.

    diffu-run --jobs run.jsonl --out /outputs/run --model /models/moss --draw-settings draw.json --gpus 8

``run.jsonl`` holds one :class:`diffu_page.jobs.PageJob` per line, each with its own ``id``; ``draw.json`` the
:class:`diffu_page.pipeline.DrawSettings` the pages are drawn with, whole (a field it leaves out keeps its default).
Ray Data reads the run file, plans each page on CPU workers (an opening becomes its two pages), draws the pages with
``--gpus`` times ``--actors-per-gpu`` actors (a ``map_batches`` actor pool) and the driver writes them to
``<out>/pages.lance`` (:mod:`diffu_runner.table`) every ``--commit-every`` pages; ``<out>/runs/`` keeps each run's
settings and counts. A run into an ``--out`` that holds a table resumes it: the pages drawn are skipped and those that
failed are drawn again. A page is named and seeded by its id, as ``diffu-generate`` names and seeds it, so a page comes
out the same from either. Everything that can be checked before the cluster loads a model is checked first
(:class:`RunRefused`). docs/INFERENCE.md is the guide.
"""

from __future__ import annotations

import datetime
import json
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Annotated, Self

import lance_ray
import ray
import typer
from huggingface_hub import try_to_load_from_cache
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from ray.data import Dataset

from diffu_page.cli.options import Loras, lora_dirs
from diffu_page.jobs import PageJob
from diffu_page.output import console, validation_problems
from diffu_page.pipeline import DrawSettings, model_family
from diffu_page.settings import settings
from diffu_runner.stages import Attention, DrawActor, DrawConfig, JudgeModel, plan_rows
from diffu_runner.table import FILE_FORMAT, PAGES, SCHEMA, drawn_ids, drop_failed, run_counts

app = typer.Typer(add_completion=False, rich_markup_mode="rich")

INDEX = "modular_model_index.json"
RUNS = "runs"  # <out>/runs/<run>.json: each run's settings and counts


class RunRefused(ValueError):
    """A run that cannot start as asked: a run file, model, judge or ``--out`` that does not go with it."""


class RunConfig(BaseModel):
    """A run: how its pages are drawn, on how many GPUs and how many actors share each, and how many drawn pages go
    into each commit to its table (a crash loses the pages drawn since the last).

    Attributes:
        gpus: The GPUs to draw on; 0 draws on the CPU (tests), with ``actors_per_gpu`` actors.
        actors_per_gpu: How many actors, each holding the whole pipeline, share a GPU.
        cpus_per_actor: The CPUs Ray keeps for each actor; the pages are planned on the rest. An actor stage that
            names none holds a whole CPU anyway while Ray Data counts it as none (Ray 2.59), which can leave the
            planning with no CPU at all.
        attention: The kernel the models' attention runs on (``auto``: PyTorch's pick): a way to rule a kernel in or
            out, not a draw setting, so a resumed run may change it.
    """

    model_config = ConfigDict(frozen=True)

    draw: DrawConfig
    gpus: int = Field(ge=0)
    actors_per_gpu: int = Field(default=1, ge=1)
    cpus_per_actor: float = Field(default=1.0, ge=0)
    commit_every: int = Field(default=64, ge=1)
    attention: Attention = Attention.AUTO

    @model_validator(mode="after")
    def _draws_where_it_runs(self) -> Self:
        on_cpu = self.draw.settings.device == "cpu"
        if on_cpu != (self.gpus == 0):
            device = self.draw.settings.device
            raise ValueError(f"a run on {device!r} with {self.gpus} GPUs: draw on the CPU with 0 GPUs, else on GPUs")
        return self

    @property
    def actors(self) -> int:
        """How many actors draw the run's pages."""
        return max(self.gpus, 1) * self.actors_per_gpu

    @property
    def gpus_per_actor(self) -> float:
        """The share of a GPU each actor takes."""
        return 0.0 if self.gpus == 0 else 1 / self.actors_per_gpu


class RunMeta(BaseModel):
    """``<out>/runs/<run>.json``: a run's settings, its run file and how many pages it holds, how many pages the table
    held drawn when it started and how many failed ones it draws again, and, once it has ended, how many it drew,
    how many failed and the median time a page took on an actor."""

    run: str
    config: RunConfig
    jobs: str
    pages: int
    in_table: int
    retried: int
    started: str
    finished: str | None = None
    drawn: int | None = None
    failed: int | None = None
    median_seconds: float | None = None


def read_pages(jobs: Path) -> list[PageJob]:
    """A run's pages: one :class:`PageJob` per line of ``jobs`` (blank lines skipped), each with an id of its own.

    Raises:
        RunRefused: If a line is not a page job or has no id, two lines share an id, or there is no page at all.
    """
    pages: list[PageJob] = []
    for number, line in enumerate(jobs.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            page = PageJob.model_validate_json(line)
        except ValidationError as error:
            raise RunRefused(f"{jobs}:{number} is not a page job: {validation_problems(error, 'line')}") from error
        if page.id is None:
            raise RunRefused(f"{jobs}:{number} has no id: a run names, seeds and resumes its pages by their ids")
        pages.append(page)
    if not pages:
        raise RunRefused(f"{jobs} holds no page jobs")
    shared = [page_id for page_id, count in Counter(page.id for page in pages).items() if count > 1]
    if shared:
        raise RunRefused(
            f"{jobs}: an id names one page, but {len(shared)} name several: {', '.join(map(str, shared[:5]))}"
        )
    return pages


def read_draw_settings(path: Path | None) -> DrawSettings:
    """The draw settings in the JSON object at ``path``, any :class:`DrawSettings` field by name (a field it leaves out
    keeps its default); the defaults when there is no file. The run hands them to its actors whole, so a field the page
    pipeline adds reaches them unchanged.

    Raises:
        RunRefused: If the file is not a JSON object of draw settings, naming a key that is not one.
    """
    if path is None:
        return DrawSettings()
    try:
        given = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunRefused(f"{path} is not a JSON object of draw settings: {error}") from error
    if not isinstance(given, dict):
        raise RunRefused(f"{path} is not a JSON object of draw settings")
    unknown = sorted(set(given) - set(DrawSettings.model_fields))
    if unknown:
        raise RunRefused(f"{path} names draw settings that do not exist: {', '.join(unknown)}")
    try:
        return DrawSettings.model_validate(given)
    except ValidationError as error:
        raise RunRefused(f"{path}: {validation_problems(error, 'the settings')}") from error


def check_settings(draw_settings: DrawSettings) -> None:
    """Check the run can draw with ``draw_settings``: each page is drawn once, so with no repair passes (a page's lines
    the judge rejects are tagged unclear; a repair is a stage of its own, still to come).

    Raises:
        RunRefused: If they ask for repair passes.
    """
    if draw_settings.repair_passes:
        raise RunRefused(
            f"repair_passes is {draw_settings.repair_passes}: a run draws each page once and tags the lines its judge "
            "rejects unclear (diffu-generate --repair draws repair passes on one GPU)"
        )


def model_index(model: Path) -> dict[str, object]:
    """The ``modular_model_index.json`` of the model directory ``model``.

    Raises:
        RunRefused: If the directory has no model index, or its blocks are no model family's.
    """
    try:
        model_family(model)
    except ValueError as error:
        raise RunRefused(str(error)) from error
    return json.loads((model / INDEX).read_text(encoding="utf-8"))


def component_folders(index: dict[str, object]) -> dict[str, Path]:
    """The local folders a model index loads its components from, by component (components from the Hub left out)."""
    folders: dict[str, Path] = {}
    for name, entry in index.items():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict):
            source = Path(str(entry[2]["pretrained_model_name_or_path"]))
            if source.is_absolute():
                folders[name] = source / str(entry[2].get("subfolder") or "")
    return folders


def check_model(model: Path) -> dict[str, object]:
    """The index of the model directory ``model``, once every component folder it names is here.

    Raises:
        RunRefused: If it is no model directory, or names a folder that is not on this machine (a model directory
            names its folders by absolute path, so it is written where it runs).
    """
    index = model_index(model)
    missing = {name: folder for name, folder in component_folders(index).items() if not folder.is_dir()}
    if missing:
        listed = ", ".join(f"{name} at {folder}" for name, folder in missing.items())
        raise RunRefused(
            f"{model} names component folders that are not on this machine ({listed}): write the model directory "
            "where it runs, with python -m diffu_moss.init --base <Base snapshot> --transformer <DiT> --out <dir>"
        )
    return index


def check_judge(judge: JudgeModel) -> None:
    """Check the judge's model is here: a local directory, or its Hub revision in the Hugging Face cache (a run never
    downloads it).

    Raises:
        RunRefused: If it is neither.
    """
    local = Path(judge.model)
    if local.is_dir():
        if not (local / "config.json").is_file():
            raise RunRefused(f"the HTR model directory {local} has no config.json")
        return
    if not isinstance(try_to_load_from_cache(judge.model, "config.json", revision=judge.revision), str):
        raise RunRefused(
            f"the HTR model {judge.model} at revision {judge.revision} is neither a local directory nor in the "
            "Hugging Face cache: stage it (hf download --revision) or pass --htr-model <dir>"
        )


def check_pages(pages: list[PageJob], draw: DrawConfig, index: dict[str, object]) -> None:
    """Check every page can be drawn as ``draw`` draws it: a reference image only for a model whose ``index`` lets it
    take one, one for every page under reference guidance, and every template and reference image named by absolute
    path (the nodes that plan and draw a page read them, from working directories of their own) and readable here.

    Raises:
        RunRefused: If any cannot.
    """
    referenced = [page for page in pages if page.reference is not None]
    if referenced and index.get("takes_reference") is False:
        raise RunRefused(
            f"{len(referenced)} pages give a reference image (the first: {referenced[0].id}), but {draw.model} takes "
            "none: its DiT was not trained with references (takes_reference in its model index)"
        )
    if draw.settings.guides_reference and len(referenced) < len(pages):
        raise RunRefused(
            f"reference guidance needs a reference image, but {len(pages) - len(referenced)} pages give none"
        )
    files = {page.template for page in pages} | {page.reference.image for page in pages if page.reference is not None}
    relative = sorted(str(path) for path in files if not path.is_absolute())
    if relative:
        raise RunRefused(
            f"{len(relative)} templates or reference images are named by a relative path (name each by its absolute "
            f"path, the same on every node): {', '.join(relative[:5])}"
        )
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise RunRefused(
            f"{len(missing)} templates or reference images are not here (each node reads them at the same path): "
            f"{', '.join(missing[:5])}"
        )


def run_metas(out: Path) -> list[RunMeta]:
    """The runs into ``out`` so far, oldest first."""
    return [
        RunMeta.model_validate_json(path.read_text(encoding="utf-8")) for path in sorted((out / RUNS).glob("*.json"))
    ]


def check_drawn_alike(out: Path, draw: DrawConfig) -> None:
    """Check every earlier run into ``out`` drew as ``draw`` draws: one table holds pages of one model and settings.

    Raises:
        RunRefused: If one drew otherwise.
    """
    for meta in run_metas(out):
        if meta.config.draw != draw:
            raise RunRefused(
                f"{out} holds pages drawn otherwise (run {meta.run}: {out / RUNS / f'{meta.run}.json'}): resume it with "
                "the same model, draw settings, page size and judge, or give a new --out"
            )


def check_run(pages: list[PageJob], config: RunConfig, out: Path) -> None:
    """Everything about a run that can be checked before the cluster loads a model.

    Raises:
        RunRefused: If the run cannot start as asked.
    """
    check_settings(config.draw.settings)
    index = check_model(config.draw.model)
    check_judge(config.draw.judge)
    check_pages(pages, config.draw, index)
    check_drawn_alike(out, config.draw)


def write_meta(out: Path, meta: RunMeta) -> None:
    """Write ``meta`` to ``<out>/runs/<run>.json``, whole or not at all."""
    path = out / RUNS / f"{meta.run}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(meta.model_dump_json(indent=1), encoding="utf-8")
    partial.replace(path)


def plan_dataset(jobs: Path, *, drawn: frozenset[str], page_pixels: int) -> Dataset:
    """The run's planned pages: the run file read by Ray Data a line a row, each page job planned on CPU workers into a
    row for each single page the table does not hold drawn (``drawn``), planned whole
    (:func:`diffu_runner.stages.plan_rows`). Every node reads ``jobs`` at the same path."""
    return ray.data.read_text(str(jobs.absolute())).flat_map(partial(plan_rows, drawn=drawn, page_pixels=page_pixels))


def draw_dataset(planned: Dataset, config: RunConfig, *, run: str) -> Dataset:
    """The planned pages drawn by ``config.actors`` actors, as the page table's rows (an actor that crashes or whose
    device fails is restarted, and its page drawn again, up to 3 times).

    Each page is its own block: an actor hands a block on only once it has drawn all of it, and Ray Data stops a
    dataset that makes no progress for 30 minutes; one page per block also spreads the pages evenly over the actors.
    The actors' arguments hold only plain values and models of their own: Ray Data hands a stage's arguments to its
    stats actor, which would import the module of any other class among them (a ``StrEnum`` of ours: the whole page
    pipeline, ~0.7 GB on the head node).
    """
    return planned.repartition(target_num_rows_per_block=1).map_batches(
        DrawActor,
        fn_constructor_kwargs={"draw": config.draw, "run": run, "attention": config.attention.value},
        batch_size=1,
        concurrency=config.actors,
        num_gpus=config.gpus_per_actor,
        num_cpus=config.cpus_per_actor,
        max_restarts=3,
        max_task_retries=3,
    )


def run(pages: list[PageJob], config: RunConfig, out: Path, *, jobs: Path) -> RunMeta:
    """Draw every page of the run file ``jobs`` (``pages``, as :func:`read_pages` read it for the checks) that
    ``<out>/pages.lance`` does not hold drawn into it, the failed ones again, committing every ``config.commit_every``
    pages; connects to the Ray cluster ``ray.init`` finds.

    Returns:
        The run's record, as ``<out>/runs/<run>.json`` keeps it.

    Raises:
        RunRefused: If the run cannot start as asked (:func:`check_run`), before anything is drawn.
    """
    check_run(pages, config, out)
    table = out / PAGES
    drawn = drawn_ids(table)
    retried = drop_failed(table)
    started = datetime.datetime.now(datetime.UTC)
    meta = RunMeta(
        run=f"{started:%Y%m%dT%H%M%S%fZ}",
        config=config,
        jobs=str(jobs),
        pages=len(pages),
        in_table=len(drawn),
        retried=retried,
        started=started.isoformat(),
    )
    write_meta(out, meta)
    if not ray.is_initialized():
        ray.init()
    lance_ray.write_lance(
        draw_dataset(plan_dataset(jobs, drawn=drawn, page_pixels=config.draw.page_pixels), config, run=meta.run),
        str(table),
        schema=SCHEMA,
        mode="append" if table.exists() else "create",
        stream=True,  # a commit every batch_size rows, from the driver, so a crash loses at most that many pages
        batch_size=config.commit_every,
        data_storage_version=FILE_FORMAT,
    )
    counts = run_counts(table, meta.run)
    finished = datetime.datetime.now(datetime.UTC).isoformat()
    meta = meta.model_copy(update={"finished": finished, **counts.model_dump()})
    write_meta(out, meta)
    return meta


def cluster_gpus() -> int:
    """How many GPUs the Ray cluster has now (connecting to it, or starting a local one, first)."""
    if not ray.is_initialized():
        ray.init()
    return int(ray.cluster_resources().get("GPU", 0))


def summary(meta: RunMeta) -> str:
    """A finished run in a line: what it drew, how fast, and where."""
    hours = (datetime.datetime.fromisoformat(str(meta.finished)) - datetime.datetime.fromisoformat(meta.started)) / (
        datetime.timedelta(hours=1)
    )
    rate = f"{(meta.drawn or 0) / hours:.0f} pages an hour" if hours > 0 else "no time"
    per_page = f", {meta.median_seconds:.1f} s a page on an actor (median)" if meta.median_seconds is not None else ""
    return (
        f"run {meta.run}: {meta.drawn} pages drawn, {meta.failed} failed, {meta.in_table} already drawn before it; "
        f"{rate} over {hours * 60:.1f} min{per_page}"
    )


@app.command()
def main(
    jobs: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="the run's pages: one diffu_page.jobs.PageJob per line, each with its id"
        ),
    ],
    out: Annotated[
        Path,
        typer.Option(help="the run's directory: pages.lance and runs/ go there; one that holds a table is resumed"),
    ],
    model: Annotated[
        Path, typer.Option(help="the model directory: MOSS-Image's (diffu-moss-init), or FLUX.3 Klein's with --lora")
    ],
    draw_settings: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="a JSON object of diffu_page.pipeline.DrawSettings fields (steps, guidance_scale, device, ...); a "
            "field it leaves out keeps its default, and none at all draws with the defaults",
        ),
    ] = None,
    gpus: Annotated[
        int | None,
        typer.Option(min=0, help="the GPUs to draw on (default: every GPU the cluster has; 0 to draw on the CPU)"),
    ] = None,
    actors_per_gpu: Annotated[
        int, typer.Option(min=1, help="actors sharing a GPU, each holding the whole pipeline (MOSS-Image: ~35 GB)")
    ] = 1,
    cpus_per_actor: Annotated[
        float, typer.Option(min=0, help="the CPUs Ray keeps for each actor; the pages are planned on the rest")
    ] = 1.0,
    commit_every: Annotated[
        int, typer.Option(min=1, help="drawn pages per commit to the table: a crash loses at most this many")
    ] = 64,
    attention: Annotated[
        Attention, typer.Option(help="the attention kernel the models run on (auto: PyTorch's pick for each call)")
    ] = Attention.AUTO,
    page_megapixels: Annotated[
        float,
        typer.Option(min=0.25, max=4.0, help="the size a page is drawn whole at, unless its job gives page_pixels"),
    ] = 1.0,
    htr_model: Annotated[
        str, typer.Option(help="the HTR model lines are read back with: a directory, or a Hub id in the HF cache")
    ] = settings.htr_model,
    htr_revision: Annotated[
        str | None, typer.Option(help="the HTR model's pinned Hub revision ('' for a directory)")
    ] = settings.htr_revision,
    lora: Loras = None,
) -> None:
    """Draw a run file's pages across a Ray cluster into a Lance table, resuming the table if it is there."""
    try:
        pages = read_pages(jobs)
        draw = DrawConfig(
            model=model,
            loras=lora_dirs(lora),
            settings=read_draw_settings(draw_settings),
            page_pixels=round(page_megapixels * 1_000_000),
            judge=JudgeModel(model=htr_model, revision=htr_revision),
        )
        if gpus is None:
            gpus = 0 if draw.settings.device == "cpu" else cluster_gpus()
        try:
            config = RunConfig(
                draw=draw,
                gpus=gpus,
                actors_per_gpu=actors_per_gpu,
                cpus_per_actor=cpus_per_actor,
                commit_every=commit_every,
                attention=attention,
            )
        except ValidationError as error:
            raise RunRefused(validation_problems(error, "the run")) from error
        meta = run(pages, config, out, jobs=jobs)
    except RunRefused as error:
        raise typer.BadParameter(str(error)) from error
    console.print(summary(meta))


if __name__ == "__main__":
    app()
