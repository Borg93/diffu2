"""Few-step distillation of a MOSS-Image DiT by on-policy distillation: a student that draws a page in a few steps
without guidance, taught by a teacher that draws it in many with guidance.

One training sample (:func:`distillation_loss`):

1. The student draws from noise along its own few noise levels (:func:`student_levels`) to one of them, chosen at
   random, without gradients (:func:`rolled_out`): the states it is trained on are the states it will meet when it
   draws, not noised real pages.
2. From that state the teacher draws on to the student's next level in several Euler sub-steps with classifier-free
   guidance (:func:`teacher_velocity`), and the straight line from the state to where the teacher arrived is the target:
   the one velocity that takes the student there in its single step.
3. The student's velocity at the state, with gradients, is regressed onto that target (mean squared error over the
   drawn image's tokens).

The sample page, when the entry has one, goes to the student and to both of the teacher's passes, as it does in
drawing. Velocities follow the trainer's convention: a state at level ``sigma`` is ``(1 - sigma) * image + sigma *
noise`` and the DiT predicts ``noise - image``, so an Euler step to ``after`` adds ``(after - sigma) * velocity``.

:func:`distill` trains a LoRA student of a teacher DiT on a cache's prompts and sample pages (the cached target only
gives the page's size; no real page is drawn toward), and saves it as a reference training saves a LoRA, so
``diffu-moss-export-ref`` merges it into the teacher's DiT. The student is then drawn in ``student_steps`` steps at
guidance 1.

    diffu-moss-distill --teacher <DiT folder> --cache <cache> --negative <negative prompt embeds> --out <run>
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from statistics import fmean
from typing import Annotated

import torch
import typer
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional as F

from diffu_moss.modular_pipelines.moss.before_denoise import SAMPLING_SHIFT, sampling_sigmas
from diffu_moss.modular_pipelines.moss.denoise import DEMO_UNCOND_SKIP_LAYERS, GUIDANCE_SCALE
from diffu_moss.train.cache import load_sample, read_entries
from diffu_moss.train.loop import CHECKPOINTS, LOG, LoraTraining, TrainSettings, console, sample_size_of
from diffu_moss.train.records import JsonRecord, Progress
from diffu_moss.transformer import MossTransformer

DISTILL_RECORD = "distill.json"  # beside a checkpoint's LoRA: how its student is drawn

app = typer.Typer(add_completion=False)


class DistillSettings(BaseModel):
    """How a student is distilled: the steps it draws in, the shift of its noise levels, the Euler sub-steps the
    teacher takes across one student step, and the teacher's guidance (its scale, and the blocks its unconditional
    pass skips)."""

    model_config = ConfigDict(frozen=True)

    student_steps: int = Field(default=8, ge=1)
    shift: float = Field(default=SAMPLING_SHIFT, gt=0)
    teacher_substeps: int = Field(default=6, ge=1)
    guidance_scale: float = Field(default=GUIDANCE_SCALE, ge=1.0)
    uncond_skip_layers: tuple[int, ...] = DEMO_UNCOND_SKIP_LAYERS


def student_levels(settings: DistillSettings) -> list[float]:
    """The noise levels the student draws through, from 1 down to 0: the sampler's levels for its step count, then 0."""
    return [*sampling_sigmas(settings.student_steps, settings.shift), 0.0]


def _velocity(
    dit: MossTransformer,
    latents: Tensor,
    sigma: float,
    text: Tensor,
    references: Sequence[Tensor] | None,
    skip_layers: frozenset[int] = frozenset(),
) -> Tensor:
    timestep = torch.tensor([sigma * 1000], device=latents.device)
    return dit(latents, timestep, text, None, skip_layers, references=references).float()


@torch.no_grad()
def rolled_out(
    student: MossTransformer,
    noise: Tensor,
    levels: Sequence[float],
    index: int,
    text: Tensor,
    references: Sequence[Tensor] | None = None,
) -> Tensor:
    """The state the student reaches at ``levels[index]`` drawing from ``noise`` (the state at ``levels[0]``) in
    Euler steps without guidance; ``noise`` itself for index 0. No gradients are kept."""
    latents = noise.float()
    for now, after in pairwise(levels[: index + 1]):
        latents = latents + (after - now) * _velocity(student, latents, now, text, references)
    return latents


@torch.no_grad()
def teacher_velocity(
    teacher: MossTransformer,
    latents: Tensor,
    start: float,
    end: float,
    text: Tensor,
    negative: Tensor,
    settings: DistillSettings,
    references: Sequence[Tensor] | None = None,
) -> Tensor:
    """The velocity that takes ``latents`` from level ``start`` to where the teacher arrives at ``end`` in one Euler
    step: the teacher's ``settings.teacher_substeps`` guided Euler sub-steps across the interval (evenly spaced
    levels), then ``(arrived - latents) / (end - start)``. At guidance scale 1 the unconditional pass is not run."""
    skipped = frozenset(settings.uncond_skip_layers)
    arrived = latents.float()
    levels = torch.linspace(start, end, settings.teacher_substeps + 1).tolist()
    for now, after in pairwise(levels):
        velocity = _velocity(teacher, arrived, now, text, references)
        if settings.guidance_scale != 1.0:
            unconditional = _velocity(teacher, arrived, now, negative, references, skipped)
            velocity = unconditional + settings.guidance_scale * (velocity - unconditional)
        arrived = arrived + (after - now) * velocity
    return (arrived - latents.float()) / (end - start)


def distillation_loss(
    student: MossTransformer,
    teacher: MossTransformer,
    noise: Tensor,
    index: int,
    text: Tensor,
    negative: Tensor,
    settings: DistillSettings,
    references: Sequence[Tensor] | None = None,
) -> Tensor:
    """The on-policy distillation loss of one sample at the student's step ``index`` (the module's docstring): the
    mean squared error between the student's velocity at the state it drew itself to and the teacher's velocity
    across that step.

    Raises:
        ValueError: If ``index`` is not one of the student's steps.
    """
    levels = student_levels(settings)
    if not 0 <= index < len(levels) - 1:
        raise ValueError(f"step {index} is not one of the student's {len(levels) - 1}")
    state = rolled_out(student, noise, levels, index, text, references)
    target = teacher_velocity(teacher, state, levels[index], levels[index + 1], text, negative, settings, references)
    return F.mse_loss(_velocity(student, state, levels[index], text, references), target)


class DistillRun(BaseModel):
    """A distillation run: its length, the student LoRA's shape and learning rate (warmed up over ``warmup`` steps),
    the samples whose losses one optimizer step averages, how the student is taught (``distill``), and when it is
    saved."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=1000, ge=1)
    rank: int = Field(default=64, ge=1)
    alpha: int = Field(default=64, ge=1)
    learning_rate: float = 1e-4
    warmup: int = Field(default=50, ge=1)
    samples_per_step: int = Field(default=1, ge=1)
    grad_clip: float = 1.0
    distill: DistillSettings = DistillSettings()
    save_every: int = Field(default=250, ge=1)
    seed: int = 0
    device: str = "cuda"


class DistillRecord(JsonRecord):
    """How a distilled student is drawn (beside its checkpoint's LoRA): the teacher it is a LoRA of, and its steps and
    noise levels' shift, at guidance 1."""

    teacher: str
    student_steps: int
    shift: float
    guidance_scale: float = 1.0


class DistillLog(BaseModel):
    """One optimizer step: the mean loss of its samples, the learning rate it stepped at, the gradients' norm before
    clipping, each sample's pair and the student step it was taught at, and its time."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    pairs: list[str]
    indices: list[int]
    seconds: float


def distill(teacher: Path, cache: Path, negative: Path, out: Path, run: DistillRun) -> Path:
    """A LoRA student of the DiT at ``teacher`` distilled on the prompts and sample pages of ``cache`` (the module's
    docstring), the teacher guided against the negative prompt's embeddings in ``negative`` (a safetensors file's
    ``prompt_embeds``); the log and checkpoints go to ``out``. Each sample draws an entry, its noise and the student
    step it is taught at from one generator seeded with ``run.seed``.

    Returns:
        The last checkpoint's directory.

    Raises:
        ValueError: If ``out`` holds a run's log already, or the cache holds no entry.
    """
    if (out / LOG).exists():
        raise ValueError(f"{out} holds a training already ({LOG}); give a new directory")
    entries = read_entries(cache)
    if not entries:
        raise ValueError(f"{cache} holds no entry to distill on")
    settings = TrainSettings(
        steps=run.steps,
        rank=run.rank,
        alpha=run.alpha,
        learning_rate=run.learning_rate,
        warmup=run.warmup,
        save_every=run.save_every,
        seed=run.seed,
        device=run.device,
    )
    training = LoraTraining(teacher, settings, sample_size_of(cache, None))
    frozen = MossTransformer.from_pretrained(str(teacher), torch_dtype=torch.bfloat16).to(run.device).eval()
    frozen.requires_grad_(False)
    negative_embeds = load_file(str(negative), device=run.device)["prompt_embeds"]
    record = DistillRecord(
        teacher=str(teacher.resolve()), student_steps=run.distill.student_steps, shift=run.distill.shift
    )
    generator = torch.Generator().manual_seed(run.seed)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out
    for step in range(1, run.steps + 1):
        started = time.monotonic()
        losses, pairs, indices = [], [], []
        for _ in range(run.samples_per_step):
            entry = entries[int(torch.randint(len(entries), (1,), generator=generator))]
            index = int(torch.randint(run.distill.student_steps, (1,), generator=generator))
            sample = load_sample(cache, entry, run.device)
            noise = torch.randn(sample.target.shape, generator=generator).to(run.device)
            references = None if sample.reference is None else [sample.reference]
            loss = distillation_loss(
                training.transformer,
                frozen,
                noise,
                index,
                sample.prompt_embeds,
                negative_embeds,
                run.distill,
                references,
            )
            (loss / run.samples_per_step).backward()
            losses.append(loss.item())
            pairs.append(entry.pair)
            indices.append(index)
        grad_norm = torch.nn.utils.clip_grad_norm_(training.parameters, run.grad_clip).item()
        learning_rate = training.step()
        logged = DistillLog(
            step=step,
            loss=fmean(losses),
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            pairs=pairs,
            indices=indices,
            seconds=time.monotonic() - started,
        )
        with (out / LOG).open("a", encoding="utf-8") as log:
            log.write(f"{logged.model_dump_json()}\n")
        if step % 10 == 0 or step == 1:
            console.print(f"step {step}: loss {logged.loss:.4f}, {logged.seconds:.1f} s")
        if step % run.save_every == 0 or step == run.steps:
            checkpoint = out / CHECKPOINTS / f"step_{step:06d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            training.save(
                checkpoint, Progress(step=step, settings=run.model_dump_json(), generator=b"", order=[], drawn=0)
            )
            record.write(checkpoint / DISTILL_RECORD)
    return checkpoint


@app.command()
def main(
    teacher: Annotated[Path, typer.Option(help="the teacher's DiT folder; the student is a LoRA of it")],
    cache: Annotated[Path, typer.Option(help="a diffu-moss-cache-pairs cache: its prompts and sample pages")],
    negative: Annotated[
        Path, typer.Option(help="the negative prompt's embeddings (a safetensors file holding prompt_embeds)")
    ],
    out: Annotated[Path, typer.Option(help="the run's directory (its log and checkpoints)")],
    steps: Annotated[int, typer.Option(min=1, help="optimizer steps")] = 1000,
    samples_per_step: Annotated[int, typer.Option(min=1, help="samples whose losses a step averages")] = 1,
    rank: Annotated[int, typer.Option(min=1, help="the student LoRA's rank (its alpha the same)")] = 64,
    learning_rate: Annotated[float, typer.Option(help="AdamW's, after the warmup")] = 1e-4,
    student_steps: Annotated[int, typer.Option(min=1, help="the steps the student draws in")] = 8,
    teacher_substeps: Annotated[
        int, typer.Option(min=1, help="the teacher's guided Euler sub-steps across one student step")
    ] = 6,
    guidance_scale: Annotated[float, typer.Option(min=1.0, help="the teacher's guidance scale")] = GUIDANCE_SCALE,
    save_every: Annotated[int, typer.Option(min=1, help="a checkpoint every this many steps, and at the end")] = 250,
    seed: Annotated[int, typer.Option(help="the samples, noise, student steps and the LoRA's start")] = 0,
    device: Annotated[str, typer.Option(help="where the student and the teacher run")] = "cuda",
) -> None:
    """Distil a few-step, unguided LoRA student of a MOSS-Image DiT by on-policy distillation."""
    run = DistillRun(
        steps=steps,
        rank=rank,
        alpha=rank,
        learning_rate=learning_rate,
        samples_per_step=samples_per_step,
        distill=DistillSettings(
            student_steps=student_steps, teacher_substeps=teacher_substeps, guidance_scale=guidance_scale
        ),
        save_every=save_every,
        seed=seed,
        device=device,
    )
    try:
        checkpoint = distill(teacher, cache, negative, out, run)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    console.print(f"the student after {steps} steps -> {checkpoint}")


if __name__ == "__main__":
    app()
