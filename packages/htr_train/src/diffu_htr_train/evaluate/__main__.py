"""``python -m diffu_htr_train.evaluate`` (run with ``envs/minicpm``'s Python): the lines to score (``ood-set``,
``set-lines``), a reader's predictions and its speed (``read``), their error rates (``cer``), and a reader's
predictions as reads for Jev's pairs (``as-reads``)."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel
from rich.table import Table

from diffu_htr_train.evaluate.cer import by_subset, nfc_trimmed, trocr_normalize, whitespace
from diffu_htr_train.evaluate.lines import OOD_REPO, ood_lines, read_lines, sampled, set_lines, write_lines
from diffu_htr_train.evaluate.readers import (
    MiniCPMSettings,
    Reading,
    TrOCRSettings,
    read_minicpm,
    read_trocr,
)
from diffu_htr_train.output import console
from diffu_htr_train.sets.jev_pairs import Read

app = typer.Typer(add_completion=False, rich_markup_mode="rich")


class Reader(StrEnum):
    MINICPM = "minicpm"
    TROCR = "trocr"


class Prediction(BaseModel):
    """A line's reference and a reader's reading of it, and why the reading ended (a row of a predictions file)."""

    id: str
    subset: str
    text: str
    prediction: str
    finished: str | None = None


class Speed(BaseModel):
    """How fast a reader read its lines (``<predictions stem>.speed.json``), with the settings it read them with."""

    reader: Reader
    lines: int
    seconds: float
    lines_per_second: float
    characters_per_second: float
    settings: MiniCPMSettings | TrOCRSettings


@app.command(name="ood-set")
def ood_set(out: Annotated[Path, typer.Option(help="the lines.jsonl to write; its images go beside it")]) -> None:
    """The Swedish Lion's out-of-domain lines (downloaded from the Hub), as lines to score."""
    from huggingface_hub import snapshot_download

    cache = Path(snapshot_download(repo_id=OOD_REPO, repo_type="dataset", allow_patterns="*.parquet"))
    lines = list(ood_lines(sorted(cache.glob("*.parquet")), out.parent / f"{out.stem}_images"))
    write_lines(lines, out)
    console.print(f"{len(lines)} lines of {len({line.subset for line in lines})} volumes -> {out}")


@app.command(name="set-lines")
def set_lines_command(
    samples: Annotated[Path, typer.Argument(exists=True, help="a line set (its test side: <stem>.test.jsonl)")],
    out: Annotated[Path, typer.Option(help="the lines.jsonl to write")],
    sample: Annotated[int | None, typer.Option(help="take only this many lines, drawn by --seed")] = None,
    seed: Annotated[int, typer.Option(help="the draw's seed")] = 0,
) -> None:
    """The lines of one of our line sets (or a draw of them), as lines to read or score."""
    lines = sampled(list(set_lines(samples)), sample, seed)
    write_lines(lines, out)
    console.print(f"{len(lines)} lines -> {out}")


@app.command()
def read(
    lines_path: Annotated[Path, typer.Option("--lines", exists=True, help="the lines.jsonl to read")],
    reader: Annotated[Reader, typer.Option(help="minicpm (through a vLLM server) or trocr (transformers)")],
    model: Annotated[str, typer.Option(help="minicpm: the served model's name; trocr: a repo or folder")],
    out: Annotated[Path, typer.Option(help="the predictions to write (JSON lines), its speed in <stem>.speed.json")],
    endpoint: Annotated[str, typer.Option(help="minicpm: the vLLM server")] = "http://localhost:8000",
    concurrency: Annotated[int, typer.Option(help="minicpm: requests at a time")] = 32,
    max_tokens: Annotated[
        int, typer.Option(help="minicpm: new tokens at most (512 a line; OpenBMB's SDK gives a text region 4,096)")
    ] = 512,
    batch: Annotated[int, typer.Option(help="trocr: lines a batch")] = 32,
    beams: Annotated[int, typer.Option(help="trocr: beams (1: greedy)")] = 1,
    dtype: Annotated[Literal["float32", "bfloat16"], typer.Option(help="trocr: the weights' dtype")] = "float32",
    limit: Annotated[int | None, typer.Option(help="read only the first this many lines")] = None,
) -> None:
    """A reader's prediction for each line, and how fast it read them (lines and characters a second)."""
    lines = read_lines(lines_path)[:limit]
    settings: MiniCPMSettings | TrOCRSettings
    reading: Reading
    if reader is Reader.MINICPM:
        settings = MiniCPMSettings(endpoint=endpoint, model=model, concurrency=concurrency, max_tokens=max_tokens)
        reading = read_minicpm(lines, settings)
    else:
        settings = TrOCRSettings(model=model, batch=batch, beams=beams, dtype=dtype)
        reading = read_trocr(lines, settings)
    rows = [
        Prediction(id=line.id, subset=line.subset, text=line.text, prediction=prediction, finished=finished)
        for line, prediction, finished in zip(lines, reading.predictions, reading.finished, strict=True)
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{row.model_dump_json()}\n" for row in rows), encoding="utf-8")
    speed = Speed(
        reader=reader,
        lines=len(lines),
        seconds=round(reading.seconds, 2),
        lines_per_second=round(len(lines) / reading.seconds, 2),
        characters_per_second=round(sum(map(len, reading.predictions)) / reading.seconds, 1),
        settings=settings,
    )
    out.with_name(f"{out.stem}.speed.json").write_text(speed.model_dump_json(indent=1) + "\n", encoding="utf-8")
    console.print(speed.model_dump_json())


def _predictions(path: Path) -> list[Prediction]:
    with path.open(encoding="utf-8") as rows:
        return [Prediction.model_validate_json(row) for row in rows if row.strip()]


@app.command()
def cer(
    predictions: Annotated[list[Path], typer.Argument(exists=True, help="predictions files (from read)")],
    subsets: Annotated[bool, typer.Option(help="a row for each subset too")] = False,
) -> None:
    """Each predictions file's error rates as each side scores itself: the Swedish Lion's per-line mean CER (whitespace
    normalised), OpenBMB's corpus CER and WER (NFC, ends trimmed), the per-line mean CER with the baseline's training
    normalisation on both sides, and the lines cut off at the token limit."""
    for path in predictions:
        rows = _predictions(path)
        triples = [(row.subset, row.text, row.prediction) for row in rows]
        lion, openbmb = by_subset(triples, whitespace), by_subset(triples, nfc_trimmed)
        normalised = by_subset(triples, trocr_normalize)
        cut: dict[str, int] = {"all": 0}
        for row in rows:
            if row.finished == "length":
                cut["all"] += 1
                cut[row.subset] = cut.get(row.subset, 0) + 1
        table = Table(title=str(path))
        columns = ("subset", "lines", "mean CER", "CER (OpenBMB)", "WER (OpenBMB)", "mean CER, norm.", "cut off")
        for column in columns:
            table.add_column(column, justify="left" if column == "subset" else "right")
        for subset in ["all", *(name for name in lion if name != "all")] if subsets else ["all"]:
            table.add_row(
                subset,
                str(lion[subset].lines),
                f"{lion[subset].mean_cer:.2%}",
                f"{openbmb[subset].corpus_cer:.2%}",
                f"{openbmb[subset].corpus_wer:.2%}",
                f"{normalised[subset].mean_cer:.2%}",
                str(cut.get(subset, 0)),
            )
        console.print(table)


@app.command(name="as-reads")
def as_reads(
    predictions: Annotated[Path, typer.Argument(exists=True, help="a predictions file (from read)")],
    reader: Annotated[str, typer.Option(help="the reader's name, as Jev's pairs record it")],
    out: Annotated[Path, typer.Option(help="the reads to write (htr-train jev-set --reads)")],
) -> None:
    """A reader's predictions as reads of the lines (id, reader, text), for ``htr-train jev-set --reads``; a line cut
    off at the token limit left out."""
    reads = [
        Read(id=row.id, reader=reader, text=row.prediction)
        for row in _predictions(predictions)
        if row.finished != "length"
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{read.model_dump_json()}\n" for read in reads), encoding="utf-8")
    console.print(f"{len(reads)} reads -> {out}")


if __name__ == "__main__":
    app()
