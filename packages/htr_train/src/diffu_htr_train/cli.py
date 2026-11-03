"""``htr-train``: train our page and line readers from run specs, and build the sets they train on.

One app with subcommands rather than the repo's one ``diffu-*`` script a command: the commands share one subject (a
run spec and its sets) and one set of machine settings, and are asked for as ``htr-train render|launch|smoke``.

- ``render SPEC``: the run checked and its commands and scripts printed; nothing written or started.
- ``launch SPEC --where spark|slurm``: the run's folder written (spec, manifest, scripts) and the run started (on the
  Spark under its memory guard; on Slurm submitted by sbatch); ``--queue`` (the Spark only) writes it and adds it to
  a queue script in ``~/diffu_queue`` instead of starting it.
- ``smoke [CONFIGS]``: every smoke spec planned for the Spark, written, and queued one after another in
  ``~/diffu_queue/htr_smoke.sh`` (not started).
- ``split-dev SET``: a dev split of whole volumes.
- ``jev-set --lines SET``: Jev's pairs from a line set's training lines (``--side test``: its test lines, to evaluate
  on).
- ``serve-target SPEC``: the vLLM command that serves a DSpark spec's target for its responses.
- ``dspark-set --prompts SET --target DIR --endpoint URL``: a DSpark draft's responses, from its served target.

    htr-train render packages/htr_train/configs/smoke/minicpm_lines.yaml
    htr-train launch packages/htr_train/configs/full/minicpm_lines_scaled.yaml --where slurm
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.syntax import Syntax

from diffu_htr_train.jobs import DSparkJob, JobRefused, RunSpec, Where, load_run_spec
from diffu_htr_train.launch import Plan, plan_run, queue_script, shell, write_run
from diffu_htr_train.output import console, err_console, validation_problems
from diffu_htr_train.records import skipped_path
from diffu_htr_train.render import RunPaths, render_responses_server
from diffu_htr_train.sets import SetRefused
from diffu_htr_train.sets.dev_split import split_dev, volumes_of
from diffu_htr_train.sets.dspark import ResponsesSettings, write_responses
from diffu_htr_train.sets.jev_pairs import PairSettings, Side, build_pairs
from diffu_htr_train.settings import Settings

app = typer.Typer(add_completion=False, rich_markup_mode="rich", no_args_is_help=True)
SMOKE_CONFIGS = Path(__file__).parents[2] / "configs" / "smoke"

SpecArgument = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="a run spec (YAML)")]
WhereOption = Annotated[Where, typer.Option(help="spark (one GPU, under the memory guard) or slurm")]


def _refused(error: Exception) -> typer.Exit:
    """A refusal printed; the command's failure to raise."""
    err_console.print(f"refused: {error}")
    return typer.Exit(1)


def _settings() -> Settings:
    """The machine's settings (``HTR_TRAIN_*``), read once a command; a setting that does not validate printed and the
    command failed."""
    try:
        return Settings()
    except ValidationError as error:
        problems = validation_problems(error, "settings")
        raise _refused(JobRefused(f"the HTR_TRAIN_* settings are not valid: {problems}")) from error


def _load(spec_path: Path) -> RunSpec:
    """The run spec at ``spec_path``; a refusal printed and the command failed."""
    try:
        return load_run_spec(spec_path)
    except JobRefused as error:
        raise _refused(error) from error


def _plan(spec_path: Path, where: Where, settings: Settings) -> Plan:
    """The run of the spec at ``spec_path`` planned ``where``; a refusal printed and the command failed."""
    try:
        return plan_run(_load(spec_path), where, settings)
    except JobRefused as error:
        raise _refused(error) from error


def _write(plan: Plan) -> Path:
    """The run's folder written; a refusal printed and the command failed."""
    try:
        return write_run(plan)
    except JobRefused as error:
        raise _refused(error) from error


def _queue(plans: list[Plan], name: str, settings: Settings) -> Path:
    """A queue script of ``plans`` written as ``<queue>/<name>.sh``."""
    settings.queue.mkdir(parents=True, exist_ok=True)
    path = settings.queue / f"{name}.sh"
    path.write_text(queue_script(plans), encoding="utf-8")
    path.chmod(0o755)
    return path


@app.command()
def render(spec: SpecArgument, where: WhereOption = Where.SPARK) -> None:
    """Print the run's commands and scripts; nothing is written or started."""
    plan = _plan(spec, where, _settings())
    for name, text in plan.scripts.items():
        console.rule(str(plan.paths.root / name))
        console.print(Syntax(text, "bash", word_wrap=True))
    console.print(f"start: {' '.join(plan.start)}")


@app.command()
def launch(
    spec: SpecArgument,
    where: WhereOption = Where.SPARK,
    queue: Annotated[
        bool, typer.Option(help="write the run and a queue script for it in ~/diffu_queue, but do not start it")
    ] = False,
) -> None:
    """Write the run's folder and start the run (or queue it)."""
    if queue and where is not Where.SPARK:
        raise _refused(JobRefused("--queue writes the Spark's guarded queue: on slurm, sbatch queues the run itself"))
    settings = _settings()
    plan = _plan(spec, where, settings)
    root = _write(plan)
    console.print(f"{plan.spec.name}: {root}")
    if queue:
        console.print(f"queued: {_queue([plan], f'htr_{plan.spec.name}', settings)}")
        return
    try:
        status = subprocess.run(plan.start, check=False).returncode
    except FileNotFoundError as error:
        refusal = JobRefused(
            f"{plan.start[0]} is not on this machine: the run is written in {root}; start it where {plan.start[0]} "
            f"is ({shlex.join(plan.start)})"
        )
        raise _refused(refusal) from error
    if status:
        err_console.print(f"{plan.spec.name} ended with exit status {status} (see {root})")
        raise typer.Exit(status)


@app.command()
def smoke(
    configs: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="the folder of smoke specs")
    ] = SMOKE_CONFIGS,
) -> None:
    """Plan and write every smoke spec for the Spark and queue them one after another (not started)."""
    settings = _settings()
    specs = sorted(configs.glob("*.yaml"))
    if not specs:
        err_console.print(f"{configs} holds no run specs")
        raise typer.Exit(1)
    plans = [_plan(spec, Where.SPARK, settings) for spec in specs]
    for plan in plans:
        console.print(f"{plan.spec.name}: {_write(plan)}")
    console.print(f"queued {len(plans)} runs: {_queue(plans, 'htr_smoke', settings)}")


@app.command(name="split-dev")
def split_dev_command(
    set_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="a training set with its meta")],
    share: Annotated[
        float, typer.Option(min=0.001, max=0.5, help="the share of its volumes held out (more than none)")
    ] = 0.03,
    seed: Annotated[int, typer.Option(help="the seed the held-out volumes are drawn by")] = 0,
    like: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="hold out the volumes of this dev set instead (another set a run mixes with this one)",
        ),
    ] = None,
) -> None:
    """Split a training set into <stem>.train.jsonl and <stem>.dev.jsonl by whole volumes."""
    try:
        split = split_dev(set_path, share=share, seed=seed, like=volumes_of(like) if like else None)
    except SetRefused as error:
        raise _refused(error) from error
    console.print(f"{split.train_samples} samples -> {split.train}")
    console.print(f"{split.dev_samples} samples of {len(split.dev_volumes)} volumes -> {split.dev}")


@app.command(name="jev-set")
def jev_set(
    lines: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="a diffu-reader-set --task lines set")],
    out: Annotated[Path, typer.Option(help="the pair set to write; its meta, crops and renders go beside it")],
    side: Annotated[
        Side, typer.Option(help="train (pairs to train on, of training lines) or test (to evaluate on, of test lines)")
    ] = Side.TRAIN,
    reads: Annotated[
        list[Path] | None,
        typer.Option(exists=True, dir_okay=False, help="a file of readers' reads of the lines (id, reader, text)"),
    ] = None,
    negatives: Annotated[int, typer.Option(min=0, help="wrong candidates a line (rule perturbations, a swap)")] = 1,
    height: Annotated[int, typer.Option(min=0, help="the height crops are scaled to (0: as cut)")] = 48,
    seed: Annotated[int, typer.Option(help="the seed each line's candidates are drawn by")] = 0,
) -> None:
    """Build Jev's pairs from a line set's training (or test) lines."""
    settings = _settings()
    if not settings.jev_font.is_file():
        err_console.print(f"the renderer's font {settings.jev_font} is missing: set HTR_TRAIN_JEV_FONT")
        raise typer.Exit(1)
    pair_settings = PairSettings(
        side=side, negatives=negatives, height=height or None, seed=seed, font=settings.jev_font
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        built = build_pairs(lines, out, settings=pair_settings, reads=reads or [])
    except SetRefused as error:
        raise _refused(error) from error
    console.print(f"{built.positives} positive and {built.negatives} negative pairs -> {out}")
    console.print(", ".join(f"{count} {origin}" for origin, count in sorted(built.origins.items())))
    if built.skipped:
        console.print(f"left out: {built.skipped} lines ({skipped_path(out)})")


@app.command(name="serve-target")
def serve_target(spec: SpecArgument) -> None:
    """Print the vLLM command that serves a DSpark spec's target for htr-train dspark-set (run it under the guard)."""
    settings = _settings()
    loaded = _load(spec)
    job = loaded.job
    if not isinstance(job, DSparkJob):
        raise _refused(JobRefused(f"{spec} is a {job.kind} run: only a dspark run serves a target"))
    paths = RunPaths(root=settings.runs.resolve() / loaded.name, envs=settings.envs.resolve(), trackio_project="")
    console.print(shell(render_responses_server(job, paths)), soft_wrap=True)


@app.command(name="dspark-set")
def dspark_set(
    prompts: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="a line or element set to prompt with")],
    target: Annotated[Path, typer.Option(exists=True, file_okay=False, help="the target model's folder, as served")],
    endpoint: Annotated[str, typer.Option(help="the vLLM server serving the target")] = "http://localhost:8000",
    out: Annotated[Path, typer.Option(help="the responses to write (speculators conversations)")] = Path(
        "data/htr/dspark_responses.jsonl"
    ),
    max_tokens: Annotated[int, typer.Option(min=1, help="an answer's length at most")] = 512,
    limit: Annotated[int | None, typer.Option(min=1, help="prompt only this many samples")] = None,
    max_failed: Annotated[float, typer.Option(min=0.0, max=0.99, help="the share of prompts that may fail")] = 0.01,
) -> None:
    """Answer a set's prompts with the served target and write them as a draft's training data."""
    try:
        settings = ResponsesSettings(
            target=target.resolve(), endpoint=endpoint, max_tokens=max_tokens, limit=limit, max_failed=max_failed
        )
    except ValidationError as error:
        raise _refused(JobRefused(validation_problems(error, "settings"))) from error
    try:
        made = write_responses(prompts, out, settings)
    except SetRefused as error:
        raise _refused(error) from error
    console.print(f"{made.answered} responses -> {out}")
    if made.failed:
        console.print(f"failed: {made.failed} ({skipped_path(out)})")


if __name__ == "__main__":
    app()
