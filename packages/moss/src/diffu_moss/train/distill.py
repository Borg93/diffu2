"""Few-step distillation of a MOSS-Image DiT by on-policy distillation: a student that draws a page in a few steps
without guidance, taught by a teacher that draws it in many with guidance.

One training sample (:func:`distillation_loss`):

1. The student draws from noise along its own few noise levels (:func:`student_levels`) to one of them, chosen at
   random, without gradients (:func:`rolled_out`): the states it is trained on are the states it will meet when it
   draws, not noised real pages.
2. From that state the teacher draws on to the student's next level through its own noise levels that lie between
   (:func:`teacher_levels`: the levels of its many-step drawing, so the target is what that drawing would do from
   there), in Euler steps with classifier-free guidance (:func:`teacher_velocity`), and the straight line from the
   state to where the teacher arrived is the target: the one velocity that takes the student there in its single step.
3. The student's velocity at the state, with gradients, is regressed onto that target (mean squared error over the
   drawn image's tokens), and with ``repulsion`` pushed from the teacher's unguided velocity across the step
   (:func:`teacher_velocities`).

The sample page, when the entry has one, goes to the student and to both of the teacher's passes, as it does in
drawing. Velocities follow the trainer's convention: a state at level ``sigma`` is ``(1 - sigma) * image + sigma *
noise`` and the DiT predicts ``noise - image``, so an Euler step to ``after`` adds ``(after - sigma) * velocity``.

:func:`distill` trains a LoRA student of a teacher DiT on a cache's prompts and sample pages (the cached target only
gives the page's size; no real page is drawn toward), and saves it as a reference training saves a LoRA. Its LoRA
also adapts the layers that read the noise level (``DISTILL_TARGETS``): a student steps much further from a level than
its teacher does. The student is drawn in ``student_steps`` steps at guidance 1 with its LoRA beside the teacher's
weights (:func:`student_of`), not merged into them: a short distillation's update is smaller than bfloat16's rounding
step for most weights, and a merge into bfloat16 drops it there.

    diffu-moss-distill --teacher <DiT folder> --cache <cache> --negative <negative prompt embeds> --out <run>
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import typer
from peft import set_peft_model_state_dict
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import Tensor
from torch.nn import functional as F

from diffu_moss.modular_pipelines.moss.before_denoise import SAMPLING_SHIFT, sampling_sigmas
from diffu_moss.modular_pipelines.moss.denoise import DEMO_UNCOND_SKIP_LAYERS, GUIDANCE_SCALE
from diffu_moss.train.cache import load_sample, read_entries, target_shape
from diffu_moss.train.distributed import Ranks, joined, left
from diffu_moss.train.loop import (
    CHECKPOINTS,
    LOG,
    LORA_TARGETS,
    LoraTraining,
    TrainSettings,
    add_lora,
    console,
    sample_size_of,
)
from diffu_moss.train.records import (
    ADAPTER_RECORD,
    ADAPTER_WEIGHTS,
    PROGRESS_RECORD,
    AdapterRecord,
    JsonRecord,
    LoraSettings,
    Progress,
)
from diffu_moss.transformer import MossTransformer

DISTILL_RECORD = "distill.json"  # beside a checkpoint's LoRA: how its student is drawn
OPTIMIZER_STATE = "optimizer.pt"
RESUMABLE = {"stop_at", "save_every", "device"}  # what a resumed run may change
# a reference LoRA's modules, and the layers that turn the noise level into the blocks' and the head's modulation
DISTILL_TARGETS = LORA_TARGETS + r"|time_embedding\.(0|2)|shared_block_modulation|video_head_modulation"

app = typer.Typer(add_completion=False)


class DistillSettings(BaseModel):
    """How a student is distilled: the steps it draws in, the steps of the teacher's drawing it is taught from (both
    on noise levels of one shift), the teacher's guidance (its scale, and the blocks its unconditional pass
    skips), and how strongly the student is pushed from the teacher's unguided velocity (``repulsion``, 0: not at
    all)."""

    model_config = ConfigDict(frozen=True)

    student_steps: int = Field(default=8, ge=1)
    shift: float = Field(default=SAMPLING_SHIFT, gt=0)
    teacher_steps: int = Field(default=50, ge=1)
    guidance_scale: float = Field(default=GUIDANCE_SCALE, ge=1.0)
    uncond_skip_layers: tuple[int, ...] = DEMO_UNCOND_SKIP_LAYERS
    repulsion: float = Field(default=0.0, ge=0.0, lt=1.0)


def student_levels(settings: DistillSettings) -> list[float]:
    """The noise levels the student draws through, from 1 down to 0: the sampler's levels for its step count, then 0."""
    return [*sampling_sigmas(settings.student_steps, settings.shift), 0.0]


def teacher_levels(start: float, end: float, settings: DistillSettings) -> list[float]:
    """The noise levels the teacher steps through from ``start`` to ``end``: the two, and between them every level of
    its own ``settings.teacher_steps``-step drawing that lies strictly inside."""
    inside = [level for level in sampling_sigmas(settings.teacher_steps, settings.shift) if end < level < start]
    return [start, *inside, end]


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
def teacher_velocities(
    teacher: MossTransformer,
    latents: Tensor,
    start: float,
    end: float,
    text: Tensor,
    negative: Tensor,
    settings: DistillSettings,
    references: Sequence[Tensor] | None = None,
) -> tuple[Tensor, Tensor | None]:
    """The teacher's guided and unguided velocities across the step from level ``start`` to ``end``.

    The guided one takes ``latents`` to where the teacher arrives at ``end`` in one Euler step: the teacher's guided
    Euler steps through :func:`teacher_levels`, then ``(arrived - latents) / (end - start)``. The unguided one is what
    its unconditional passes along that same path add up to across the step (each weighted by its step's share of it),
    or None where they are not run: at guidance scale 1 without repulsion.
    """
    skipped = frozenset(settings.uncond_skip_layers)
    unconditional_pass = settings.guidance_scale != 1.0 or settings.repulsion > 0
    arrived = latents.float()
    unguided = torch.zeros_like(arrived) if unconditional_pass else None
    for now, after in pairwise(teacher_levels(start, end, settings)):
        velocity = _velocity(teacher, arrived, now, text, references)
        if unguided is not None:
            unconditional = _velocity(teacher, arrived, now, negative, references, skipped)
            velocity = unconditional + settings.guidance_scale * (velocity - unconditional)
            unguided = unguided + (after - now) * unconditional
        arrived = arrived + (after - now) * velocity
    span = end - start
    return (arrived - latents.float()) / span, None if unguided is None else unguided / span


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
    step (:func:`teacher_velocities`' guided one)."""
    return teacher_velocities(teacher, latents, start, end, text, negative, settings, references)[0]


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
    across that step, less ``settings.repulsion`` times its mean squared error to the teacher's unguided velocity
    across it. That is least where the student steps at ``guided + repulsion / (1 - repulsion) * (guided - unguided)``:
    the teacher's step, pushed further along its guidance.

    Raises:
        ValueError: If ``index`` is not one of the student's steps.
    """
    levels = student_levels(settings)
    if not 0 <= index < len(levels) - 1:
        raise ValueError(f"step {index} is not one of the student's {len(levels) - 1}")
    state = rolled_out(student, noise, levels, index, text, references)
    guided, unguided = teacher_velocities(
        teacher, state, levels[index], levels[index + 1], text, negative, settings, references
    )
    velocity = _velocity(student, state, levels[index], text, references)
    loss = F.mse_loss(velocity, guided)
    if settings.repulsion > 0 and unguided is not None:
        loss = loss - settings.repulsion * F.mse_loss(velocity, unguided)
    return loss


class DistillRun(BaseModel):
    """A distillation run: its length, the student LoRA's shape (``targets``: the modules it adapts, a regex matched
    in full against the DiT's module names) and learning rate (warmed up over ``warmup`` steps),
    the samples whose losses one optimizer step averages, how the student is taught (``distill``), and when it is
    saved; ``stop_at`` ends this run at that step with a checkpoint (a chain of short jobs, each resumed from the
    last)."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=1000, ge=1)
    rank: int = Field(default=64, ge=1)
    alpha: int = Field(default=64, ge=1)
    targets: str = DISTILL_TARGETS
    learning_rate: float = 2e-5
    warmup: int = Field(default=50, ge=1)
    samples_per_step: int = Field(default=1, ge=1)
    grad_clip: float = 1.0
    distill: DistillSettings = DistillSettings()
    save_every: int = Field(default=250, ge=1)
    stop_at: int | None = Field(default=None, ge=1)
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


def distill(
    teacher: Path,
    cache: Path,
    negative: Path,
    out: Path,
    run: DistillRun,
    *,
    resume: Path | None = None,
    ranks: Ranks | None = None,
) -> Path:
    """A LoRA student of the DiT at ``teacher`` distilled on the prompts and sample pages of ``cache`` (the module's
    docstring), the teacher guided against the negative prompt's embeddings in ``negative`` (a safetensors file's
    ``prompt_embeds``); the log and checkpoints go to ``out``. Each sample draws an entry, the student step it is
    taught at and its noise from one generator seeded with ``run.seed``.

    Over several ``ranks`` (a ``torchrun`` launch) every rank holds the teacher and the student whole, draws every
    sample's entry, step and noise in turn as one process does, computes the samples it owns, and the ranks sum their
    gradients before each takes the same optimizer step: one process's training up to the sum's rounding. The main rank
    writes the log and the checkpoints. From ``resume``, a checkpoint of ``out``, the run goes on at its next step
    with its optimizer's state and its generator's, as the run would have gone on.

    Returns:
        The last checkpoint's directory.

    Raises:
        ValueError: If ``out`` holds a run's log already (unless resumed), the cache holds no entry, the samples of a
            step do not share out evenly among the ranks, or ``resume`` holds no state of this run's settings.
    """
    ranks = Ranks() if ranks is None else ranks
    if run.samples_per_step % ranks.world:
        raise ValueError(f"{run.samples_per_step} samples a step do not share out among {ranks.world} ranks")
    if resume is None and (out / LOG).exists():
        raise ValueError(f"{out} holds a training already ({LOG}); give a new directory")
    entries = read_entries(cache)
    if not entries:
        raise ValueError(f"{cache} holds no entry to distill on")
    device = run.device
    settings = TrainSettings(
        steps=run.steps,
        rank=run.rank,
        alpha=run.alpha,
        learning_rate=run.learning_rate,
        warmup=run.warmup,
        save_every=run.save_every,
        seed=run.seed,
        device=device,
    )
    lora = LoraSettings(rank=run.rank, alpha=run.alpha, target_modules=run.targets)
    training = LoraTraining(teacher, settings, sample_size_of(cache, None), lora=lora)
    frozen = MossTransformer.from_pretrained(str(teacher), torch_dtype=torch.bfloat16).to(device).eval()
    frozen.requires_grad_(False)
    negative_embeds = load_file(str(negative), device=device)["prompt_embeds"]
    record = DistillRecord(
        teacher=str(teacher.resolve()), student_steps=run.distill.student_steps, shift=run.distill.shift
    )
    generator = torch.Generator().manual_seed(run.seed)
    done = 0
    if resume is not None:
        done = _resumed(training, generator, resume, run, device)
        if ranks.main:
            lines = (out / LOG).read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [line for line in lines if DistillLog.model_validate_json(line).step <= done]
            (out / LOG).write_text("".join(kept), encoding="utf-8")
    if ranks.main:
        out.mkdir(parents=True, exist_ok=True)
    checkpoint = out if resume is None else resume
    last = run.steps if run.stop_at is None else min(run.stop_at, run.steps)
    for step in range(done + 1, last + 1):
        started = time.monotonic()
        losses = torch.zeros(run.samples_per_step, dtype=torch.float64, device=device)
        pairs, indices = [], []
        for place in range(run.samples_per_step):
            entry = entries[int(torch.randint(len(entries), (1,), generator=generator))]
            index = int(torch.randint(run.distill.student_steps, (1,), generator=generator))
            noise = torch.randn(target_shape(cache, entry), generator=generator)
            pairs.append(entry.pair)
            indices.append(index)
            if not ranks.owns(place):  # drawn all the same, so every rank's generator goes on alike
                continue
            sample = load_sample(cache, entry, device)
            references = None if sample.reference is None else [sample.reference]
            loss = distillation_loss(
                training.transformer,
                frozen,
                noise.to(device),
                index,
                sample.prompt_embeds,
                negative_embeds,
                run.distill,
                references,
            )
            (loss / run.samples_per_step).backward()
            losses[place] = loss.detach().double()
        if ranks.mesh is not None:  # each rank's share of the mean's gradient, summed
            dist.all_reduce(losses)
            for parameter in training.parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(training.parameters, run.grad_clip).item()
        learning_rate = training.step()
        if not ranks.main:
            continue
        logged = DistillLog(
            step=step,
            loss=losses.mean().item(),
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            pairs=pairs,
            indices=indices,
            seconds=time.monotonic() - started,
        )
        with (out / LOG).open("a", encoding="utf-8") as log:
            log.write(f"{logged.model_dump_json()}\n")
        if step % 10 == 0 or step == done + 1:
            console.print(f"step {step}: loss {logged.loss:.4f}, {logged.seconds:.1f} s")
        if step % run.save_every == 0 or step == last:
            checkpoint = out / CHECKPOINTS / f"step_{step:06d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            progress = Progress(
                step=step,
                settings=run.model_dump_json(exclude=RESUMABLE),
                generator=bytes(generator.get_state().numpy()),
                order=[],
                drawn=0,
            )
            training.save(checkpoint, progress)
            record.write(checkpoint / DISTILL_RECORD)
            torch.save(training.state_dict(), checkpoint / OPTIMIZER_STATE)
            progress.write(checkpoint / PROGRESS_RECORD)  # last: a checkpoint with it is whole
    return checkpoint


def _resumed(training: LoraTraining, generator: torch.Generator, checkpoint: Path, run: DistillRun, device: str) -> int:
    """``training`` and ``generator`` set to the state saved in ``checkpoint``; the step it was saved at.

    Raises:
        ValueError: If the checkpoint holds no whole state, or was trained with other settings.
    """
    if not (checkpoint / PROGRESS_RECORD).is_file():
        raise ValueError(f"{checkpoint} holds no state to resume from")
    progress = Progress.read(checkpoint / PROGRESS_RECORD)
    if progress.settings != run.model_dump_json(exclude=RESUMABLE):
        raise ValueError(f"{checkpoint} was trained with other settings: {progress.settings}")
    weights = load_file(str(checkpoint / ADAPTER_WEIGHTS), device=device)
    loaded = set_peft_model_state_dict(training.transformer, weights)
    if loaded.unexpected_keys:
        raise ValueError(f"{checkpoint}: weights that are not its LoRA's")
    training.load_state_dict(torch.load(checkpoint / OPTIMIZER_STATE, map_location=device, weights_only=True))
    generator.set_state(torch.frombuffer(bytearray(progress.generator), dtype=torch.uint8))
    return progress.step


def student_of(teacher: Path, checkpoint: Path, *, device: str = "cuda") -> MossTransformer:
    """The student saved in ``checkpoint``: the DiT at ``teacher`` in bfloat16 on ``device`` with the checkpoint's
    LoRA beside its weights in float32, as it was trained (each adapted layer adds its LoRA's output to its own), in
    eval mode. Drawn in its record's (``DISTILL_RECORD``) steps at guidance 1.

    Raises:
        ValueError: If the checkpoint's weights are not its LoRA's.
    """
    student = MossTransformer.from_pretrained(str(teacher), torch_dtype=torch.bfloat16)
    add_lora(student, AdapterRecord.read(checkpoint / ADAPTER_RECORD).lora)
    loaded = set_peft_model_state_dict(student, load_file(str(checkpoint / ADAPTER_WEIGHTS)))
    if loaded.unexpected_keys:
        raise ValueError(f"{checkpoint}: weights that are not its LoRA's")
    return student.requires_grad_(False).to(device).eval()


def latest_checkpoint(out: Path) -> Path | None:
    """The newest whole checkpoint of the run in ``out`` (its progress record is written last), or None."""
    whole = [folder for folder in sorted((out / CHECKPOINTS).glob("step_*")) if (folder / PROGRESS_RECORD).is_file()]
    return whole[-1] if whole else None


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
    learning_rate: Annotated[float, typer.Option(help="AdamW's, after the warmup")] = 2e-5,
    repulsion: Annotated[
        float,
        typer.Option(min=0.0, max=0.99, help="how strongly the student is pushed from the teacher's unguided step"),
    ] = 0.0,
    student_steps: Annotated[int, typer.Option(min=1, help="the steps the student draws in")] = 8,
    teacher_steps: Annotated[
        int, typer.Option(min=1, help="the steps of the teacher's drawing the student is taught from")
    ] = 50,
    guidance_scale: Annotated[float, typer.Option(min=1.0, help="the teacher's guidance scale")] = GUIDANCE_SCALE,
    save_every: Annotated[int, typer.Option(min=1, help="a checkpoint every this many steps, and at the end")] = 250,
    stop_at: Annotated[
        int | None, typer.Option(min=1, help="train only to this step, save a checkpoint there and stop")
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            help="a checkpoint of --out to go on from, or 'latest': its newest whole one (none: from the start)"
        ),
    ] = None,
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
            student_steps=student_steps,
            teacher_steps=teacher_steps,
            guidance_scale=guidance_scale,
            repulsion=repulsion,
        ),
        save_every=save_every,
        stop_at=stop_at,
        seed=seed,
        device=device,
    )
    ranks, device = joined(device)
    run = run.model_copy(update={"device": device})
    start = latest_checkpoint(out) if resume == "latest" else None if resume is None else Path(resume)
    try:
        checkpoint = distill(teacher, cache, negative, out, run, resume=start, ranks=ranks)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    finally:
        left()
    if ranks.main:
        console.print(f"the student -> {checkpoint}")


if __name__ == "__main__":
    app()
