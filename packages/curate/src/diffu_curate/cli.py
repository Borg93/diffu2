"""``diffu-curate``: a pool's pages scored, its labels calibrated against the ground truth, a sample staged."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from diffu_curate.calibration import OLD_PIPELINE, calibrate, write_calibration
from diffu_curate.gutter import (
    find_gutters,
    load_gutter_net,
    read_gutters,
    read_thumbnails,
    save_gutter_net,
    train_gutter_net,
    training_scans,
    write_gutters,
)
from diffu_curate.page_features import write_pool_features
from diffu_curate.pages import GROUND_TRUTH, pool_groups, truth_scans, write_pages
from diffu_curate.sample import stage_sample, truth_volumes
from diffu_curate.settings import CurateSettings, Pool
from diffu_curate.table import held_values

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)

GROUND_TRUTH_CHUNK = 100  # ground-truth scans a worker reads at a time
PoolName = Annotated[str, typer.Option("--pool", help="the pool's name: svea, andraarkiv")]


def _pool(settings: CurateSettings, name: str) -> Pool:
    try:
        return settings.pool(name)
    except ValueError as error:
        err_console.print(f"[bold red]error:[/bold red] {error}")
        raise typer.Exit(code=2) from error


@app.command()
def features(
    pool: PoolName,
    table: Annotated[Path, typer.Option(help="the Lance table of pages, appended to")],
    workers: Annotated[int, typer.Option(help="processes reading label files")] = 8,
) -> None:
    """Add every scan of a pool to the page table, with what its ALTO file alone says about its labels."""
    for done, total, rows in write_pool_features(_pool(CurateSettings(), pool), table, workers):
        console.print(f"{done} of {total} volumes, {rows} pages")
    console.print(f"[green]done[/green]: {table}")


@app.command()
def pages(
    pool: Annotated[str, typer.Option("--pool", help="a pool's name, or gt for the ground truth")],
    table: Annotated[Path, typer.Option(help="the Lance table of single pages, appended to")],
    gutters: Annotated[Path | None, typer.Option(help="the table of gutters to cut by (diffu-curate gutters)")] = None,
    workers: Annotated[int, typer.Option(help="processes reading label files")] = 8,
) -> None:
    """Add every single page of a pool, or of the ground truth, to the page table: where it is on its scan. Without
    --gutters a scan is cut by its labelled lines, as the set builder cuts it, and the row says what makes that cut
    doubtful; with it, at the gutter read from its image."""
    settings = CurateSettings()
    cuts = read_gutters(gutters, pool) if gutters else None
    if pool == GROUND_TRUTH:
        if held_values(table, "pool", f"pool = '{GROUND_TRUTH}'"):
            err_console.print(f"[bold red]error:[/bold red] {table} already holds the ground truth's pages")
            raise typer.Exit(code=2)
        scans = truth_scans(settings.ground_truth, settings.test_list, cuts)
        groups = [scans[i : i + GROUND_TRUTH_CHUNK] for i in range(0, len(scans), GROUND_TRUTH_CHUNK)]
    else:
        groups = pool_groups(_pool(settings, pool), table, cuts)
    for done, total, rows in write_pages(groups, table, workers):
        console.print(f"{done} of {total} groups, {rows} pages")
    console.print(f"[green]done[/green]: {table}")


@app.command()
def gutter_train(
    pages_table: Annotated[Path, typer.Option("--pages", help="the page table cut by lines, to learn from")],
    weights: Annotated[Path, typer.Option(help="where the network's weights are written")],
    openings: Annotated[int, typer.Option(help="openings with both pages labelled, from each pool")] = 4500,
    sheets: Annotated[int, typer.Option(help="single sheets, from each pool")] = 2000,
    held_out: Annotated[int, typer.Option(help="scans kept aside to report on")] = 1500,
    seed: Annotated[int, typer.Option(help="the draw's and the training's seed")] = 0,
    workers: Annotated[int, typer.Option(help="processes reading scans")] = 16,
) -> None:
    """Train the gutter network on the cuts the page table already knows, and say how it does on scans kept aside."""
    scans = training_scans(pages_table, openings, sheets, seed)
    thumbnails = read_thumbnails([image for image, _ in scans], workers)
    read = [(thumbnail, cut) for thumbnail, (_, cut) in zip(thumbnails, scans, strict=True) if thumbnail is not None]
    aside, train = read[:held_out], read[held_out:]
    console.print(f"{len(train)} scans to train on, {len(aside)} kept aside")
    net = train_gutter_net(np.stack([t for t, _ in train]), [cut for _, cut in train], seed=seed)
    save_gutter_net(net, weights)
    found = find_gutters(net, np.stack([t for t, _ in aside]))
    opened = [(gutter, cut) for gutter, (_, cut) in zip(found, aside, strict=True) if cut is not None]
    single = [gutter for gutter, (_, cut) in zip(found, aside, strict=True) if cut is None]
    console.print(
        f"openings kept aside: {sum(g.found for g, _ in opened) / max(1, len(opened)):.3f} found, "
        f"{sum(abs(g.x - cut) < 0.02 for g, cut in opened) / max(1, len(opened)):.3f} within 2 % of the width of "
        f"their label; single sheets given a gutter: {sum(g.found for g in single) / max(1, len(single)):.3f}"
    )
    console.print(f"[green]done[/green]: {weights}")


@app.command()
def gutters(
    pages_table: Annotated[Path, typer.Option("--pages", help="the page table naming the scans")],
    table: Annotated[Path, typer.Option(help="the Lance table of gutters, appended to")],
    weights: Annotated[Path, typer.Option(help="the gutter network's weights (diffu-curate gutter-train)")],
    workers: Annotated[int, typer.Option(help="processes reading scans")] = 16,
) -> None:
    """Read the gutter of every scan the page table names from its image, into the table of gutters."""
    for done, total in write_gutters(load_gutter_net(weights), pages_table, table, workers):
        console.print(f"{done} of {total} scans")
    console.print(f"[green]done[/green]: {table}")


@app.command()
def calibration(
    pool: PoolName,
    out: Annotated[Path, typer.Option(help="the folder its three Lance tables are written to")],
) -> None:
    """Measure a pool's old labels against the ground truth on the scans both hold."""
    settings = CurateSettings()
    result = calibrate(settings.ground_truth, _pool(settings, pool), settings.test_list)
    write_calibration(result, out)
    summary = Table(title=f"{pool}: {len(result.scans)} ground-truth scans")
    for column in ("labels by", "split", "scans"):
        summary.add_column(column)
    counts: dict[tuple[str, str], int] = {}
    for scan in result.scans:
        counts[scan.source, scan.split] = counts.get((scan.source, scan.split), 0) + 1
    for (source, split), count in sorted(counts.items()):
        summary.add_row(source, split, str(count))
    console.print(summary)
    errors = sum(line.errors for line in result.lines)
    chars = sum(len(line.truth) for line in result.lines)
    console.print(
        f"{len(result.pages)} scans labelled by {OLD_PIPELINE}, {len(result.lines)} matched lines, "
        f"CER over them {errors / max(1, chars):.3f}"
    )


@app.command()
def sample(
    pool: PoolName,
    out: Annotated[Path, typer.Option(help="the folder the sample is staged in")],
    scans: Annotated[int, typer.Option(help="how many scans to stage")],
    seed: Annotated[int, typer.Option(help="the draw's seed")] = 0,
) -> None:
    """Stage a random sample of a pool's scans for diffu-reader-set, every volume the ground truth is in left out."""
    settings = CurateSettings()
    chosen = _pool(settings, pool)
    staged = stage_sample(chosen, out, scans, seed, truth_volumes(settings.ground_truth, chosen))
    console.print(
        f"{staged.scans} scans of {staged.offered} staged in {out} "
        f"({staged.volumes_left_out} ground-truth volumes left out)"
    )
