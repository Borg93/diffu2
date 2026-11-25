"""A run's checkpoints read on dev pages: the character error rate of each, written beside the run and to trackio.

The trainer's own number at a checkpoint is the dev loss, which has not followed the error rate. This reads a fixed
draw of the dev pages (whole volumes that are in no training set, so the number is for unseen books) with every
checkpoint as it appears, and scores the readings against the ground truth: the edits over the truth's characters,
for the draw and for each volume. Run with ``envs/nemotron_parse``'s Python, on a GPU with about 10 GB free::

    CUDA_VISIBLE_DEVICES=0 packages/htr_train/envs/nemotron_parse/.venv/bin/python \\
        -m diffu_htr_train.nemotron_parse.checkpoint_cer runs/htr/RUN --dev data/htr/nemotron_parse_pages.dev.jsonl

It waits for the run's folder and for new checkpoints until ``--until`` (or the run's ``BUILT``-like end: no new
checkpoint for ``--idle`` minutes). ``RUN/dev_cer.jsonl`` gets a row a checkpoint, ``RUN/dev_reads/step-N.jsonl`` what
the checkpoint read on each page (so a number that jumps can be traced to its pages); trackio gets the run ``RUN-cer``
in the same project (a run of its own, so the trainer's run is not written to by two processes).
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from pydantic import BaseModel, ConfigDict

from diffu_htr_train.records import text_lines

LAYOUT = re.compile(r"<(?:x|y)_[0-9.]+>|<class_[A-Za-z-]+>")
TABLE = re.compile(r"\\begin\{tabular\}\{[^}]*\}|\\end\{tabular\}|\\linebreak\{\}|\\\\| & ")
MAX_NEW = 4000  # as the test read (runs/htr/verify/nemotron_parse_test_read.py)
NO_REPEAT = 10  # as the test reads named norepeat10


class DevPage(BaseModel):
    """A dev page to read: its image, the prompt the trainer gave it, the truth's text and its volume."""

    model_config = ConfigDict(frozen=True)

    id: str
    volume: str
    image: str
    prompt: str
    truth: str


class Score(BaseModel):
    """A checkpoint's error on the draw.

    Attributes:
        step: The checkpoint's step.
        cer: Edits over the truth's characters, the draw's pages together.
        volumes: The same for each volume.
        pages: How many pages were read.
        seconds: How long the reading took.
    """

    model_config = ConfigDict(frozen=True)

    step: int
    cer: float
    volumes: dict[str, float]
    pages: int
    seconds: float


class PageRead(BaseModel):
    """What a checkpoint read on one dev page, and how far that is from the truth."""

    model_config = ConfigDict(frozen=True)

    id: str
    volume: str
    chars: int
    edits: int
    reading: str


def page_text(answer: str) -> str:
    """A target or a reading as plain text: boxes, classes and table markup out, white space as one space."""
    return " ".join(TABLE.sub(" ", LAYOUT.sub("\n", answer)).replace("#", " ").split())


def edits(truth: str, reading: str) -> int:
    """The Levenshtein distance between two texts, a row of the table at a time."""
    if not truth or not reading:
        return len(truth) + len(reading)
    other = np.frombuffer(reading.encode("utf-32-le"), dtype=np.uint32)
    reach = np.arange(len(other) + 1)
    row = reach.copy()
    for k, char in enumerate(np.frombuffer(truth.encode("utf-32-le"), dtype=np.uint32), 1):
        swap = row[:-1] + (other != char)
        best = np.minimum(row[1:] + 1, swap)
        row = np.minimum.accumulate(np.concatenate(([k], best)) - reach) + reach
    return int(row[-1])


def drawn(pages: Sequence[DevPage], per_volume: int, seed: int) -> list[DevPage]:
    """At most ``per_volume`` pages of each volume, drawn by ``seed``: a large volume does not outweigh a small one."""
    by_volume: dict[str, list[DevPage]] = defaultdict(list)
    for page in pages:
        by_volume[page.volume].append(page)
    picked: list[DevPage] = []
    for volume in sorted(by_volume):
        held = sorted(by_volume[volume], key=lambda page: page.id)
        picked.extend(held if len(held) <= per_volume else random.Random(seed).sample(held, per_volume))
    return picked


def page_reads(pages: Sequence[DevPage], readings: Sequence[str]) -> list[PageRead]:
    """The readings of ``pages`` set against their truth, a page at a time."""
    return [
        PageRead(
            id=page.id, volume=page.volume, chars=len(page.truth), edits=edits(page.truth, reading), reading=reading
        )
        for page, reading in zip(pages, readings, strict=True)
    ]


def scored(step: int, reads: Sequence[PageRead], seconds: float) -> Score:
    """The error rate of the pages together and of each volume."""
    wrong: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    for page in reads:
        wrong[page.volume] += page.edits
        chars[page.volume] += page.chars
    return Score(
        step=step,
        cer=sum(wrong.values()) / max(1, sum(chars.values())),
        volumes={volume: wrong[volume] / max(1, chars[volume]) for volume in sorted(chars)},
        pages=len(reads),
        seconds=seconds,
    )


def dev_pages(samples: Path) -> list[DevPage]:
    """The pages of a page set (``diffu-reader-set``: a chat sample a page, its meta file beside it)."""
    metas = text_lines(samples.with_name(samples.name.replace(".jsonl", ".meta.jsonl")))
    pages: list[DevPage] = []
    for line, meta_line in zip(text_lines(samples), metas, strict=True):
        row, meta = json.loads(line), json.loads(meta_line)
        pages.append(
            DevPage(
                id=meta["id"],
                volume=meta["volume"],
                image=row["images"][0],
                prompt=row["messages"][0]["content"].replace("<image>", ""),
                truth=page_text(row["messages"][1]["content"]),
            )
        )
    return pages


def read(checkpoint: Path, pages: Sequence[DevPage]) -> list[str]:
    """Each page read by ``checkpoint``, decoded as the test reads are (greedy, no 10-gram twice)."""
    import torch  # heavy, and only the reading needs them
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()
    readings: list[str] = []
    for page in pages:
        with Image.open(page.image) as image:
            inputs = processor(
                images=[image.convert("RGB")], text=page.prompt, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW, no_repeat_ngram_size=NO_REPEAT)
        answer = processor.batch_decode(ids, skip_special_tokens=False)[0].split(page.prompt, 1)[-1]
        readings.append(page_text(answer.replace("</s>", "").replace("<s>", "").replace("<pad>", "")))
    del model
    torch.cuda.empty_cache()
    return readings


def unscored(run: Path, done: set[int]) -> list[tuple[int, Path]]:
    """The run's finished checkpoints (their trainer state is written last) that are not ``done``, oldest first."""
    found = [
        (int(state.parent.name.removeprefix("checkpoint-")), state.parent)
        for state in (run / "output").glob("checkpoint-*/trainer_state.json")
    ]
    return sorted((step, folder) for step, folder in found if step not in done)


def main(
    run: Annotated[Path, typer.Argument(help="the run's folder (its checkpoints are in output/)")],
    dev: Annotated[Path, typer.Option(help="the dev page set")] = Path("data/htr/nemotron_parse_pages.dev.jsonl"),
    per_volume: Annotated[int, typer.Option(help="pages of each volume at most")] = 30,
    seed: Annotated[int, typer.Option(help="the draw's seed")] = 0,
    project: Annotated[str, typer.Option(help="the trackio project; empty for none")] = "diffu",
    until: Annotated[str | None, typer.Option(help="stop at this time, 'YYYY-MM-DD HH:MM'")] = None,
    idle: Annotated[int, typer.Option(help="stop after so many minutes without a new checkpoint")] = 240,
) -> None:
    """Score every checkpoint of RUN on the dev draw, as they appear."""
    pages = drawn(dev_pages(dev), per_volume, seed)
    out = run / "dev_cer.jsonl"
    end = datetime.strptime(until, "%Y-%m-%d %H:%M").timestamp() if until else None
    tracked = None
    last = time.time()
    typer.echo(f"{len(pages)} dev pages of {len({page.volume for page in pages})} volumes; scores in {out}")
    while (end is None or time.time() < end) and time.time() - last < idle * 60:
        rows = {json.loads(line)["step"] for line in text_lines(out)} if out.exists() else set()
        done = {int(file.stem.removeprefix("step-")) for file in (run / "dev_reads").glob("step-*.jsonl")}
        waiting = unscored(run, done)
        if not waiting:
            time.sleep(60)
            continue
        step, folder = waiting[-1] if not done else waiting[0]  # the newest first on a start, then in order
        started = time.time()
        try:
            reads = page_reads(pages, read(folder, pages))
        except OSError as error:  # the trainer removed the checkpoint while it was being read
            typer.echo(f"checkpoint {step} went away: {error}")
            continue
        score = scored(step, reads, round(time.time() - started, 1))
        (run / "dev_reads").mkdir(exist_ok=True)
        (run / "dev_reads" / f"step-{step}.jsonl").write_text(
            "".join(page.model_dump_json() + "\n" for page in reads), "utf-8"
        )
        if step in rows:  # scored before its readings were kept: read again for the pages, the row stays
            last = time.time()
            continue
        with out.open("a", encoding="utf-8") as file:
            file.write(score.model_dump_json() + "\n")
        typer.echo(f"step {step}: dev CER {score.cer:.2%} on {score.pages} pages in {score.seconds:.0f} s")
        if project:
            import trackio

            tracked = tracked or trackio.init(project=project, name=f"{run.name}-cer", resume="allow")
            trackio.log(
                {"eval/dev_cer": score.cer, **{f"eval/dev_cer/{v}": c for v, c in score.volumes.items()}}, step=step
            )
        last = time.time()
    if tracked is not None:
        import trackio

        trackio.finish()


if __name__ == "__main__":
    typer.run(main)
