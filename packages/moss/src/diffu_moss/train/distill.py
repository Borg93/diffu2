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

With ``DistillRun.matching`` (``--matching``) a distribution-matching term is added to the regression: DMD's
(arXiv 2311.18828, with DMD2's two time-scale updates, arXiv 2405.14867) in this module's velocities. The student's
page at its step is where its velocity points from the state, ``page = state - sigma * velocity``, with that
velocity's gradients. The page is noised again to a level ``s`` drawn uniformly in ``matching.levels`` (``x_s = (1 -
s) * page + s * noise``), and two scores give their velocity there without gradients: the teacher, guided as it
teaches (the real score), and the fake score, a second LoRA on the teacher's DiT trained beside the student by flow
matching on the student's pages alone (:func:`fake_score_loss`). Each velocity names a page, ``x_s - s * v``, and the
student's page is pulled from the fake score's toward the teacher's (:func:`matching_loss`): where the student's
pages are distributed as the teacher's would be, the two agree and nothing is pulled. After each of the student's
optimizer steps the fake score takes ``matching.updates`` of its own on that step's pages, each at a new level and
noise. The sample page goes to both scores as to the student.
"""

from __future__ import annotations

import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Self

import torch
import torch.distributed as dist
import typer
from peft import set_peft_model_state_dict
from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    cast_lora,
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
FAKE_SCORE = "fake_score"  # in a matching run's checkpoint: its fake score's LoRA and optimizer state
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


class MatchingSettings(BaseModel):
    """A run's distribution-matching term (the module's docstring): its weight beside the regression; the teacher's
    guidance scale as the real score (None: ``DistillSettings.guidance_scale``, the scale it teaches at; 1: its
    conditional pass alone); the levels a student's page is noised to again, drawn uniformly between the two (DMD's
    0.02 to 0.98); and the fake score: its LoRA's rank (alpha the same, on the student's modules), its learning rate,
    and its optimizer steps for each of the student's (DMD2's two time-scale rule: 5)."""

    model_config = ConfigDict(frozen=True)

    weight: float = Field(default=1.0, gt=0)
    guidance_scale: float | None = Field(default=None, ge=1.0)
    levels: tuple[float, float] = (0.02, 0.98)
    rank: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    updates: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _levels_in_order(self) -> Self:
        low, high = self.levels
        if not 0 < low <= high <= 1:
            raise ValueError(f"the levels {self.levels} are not two in (0, 1], the lower first")
        return self


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
    return _taught(student, teacher, noise, index, text, negative, settings, references)[0]


def _taught(
    student: MossTransformer,
    teacher: MossTransformer,
    noise: Tensor,
    index: int,
    text: Tensor,
    negative: Tensor,
    settings: DistillSettings,
    references: Sequence[Tensor] | None,
) -> tuple[Tensor, Tensor]:
    """:func:`distillation_loss`, and the student's page at that step: where its velocity points from the state it
    drew itself to (``state - sigma * velocity``), with the velocity's gradients."""
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
    return loss, state - levels[index] * velocity


def matching_loss(page: Tensor, noised: Tensor, level: float, real: Tensor, fake: Tensor) -> Tensor:
    """DMD's distribution-matching loss of a student's ``page``, in velocities: ``real`` and ``fake`` are the real
    and the fake score's velocities at ``noised``, the page noised to ``level``, and each names a page, ``noised -
    level * velocity``. The loss's gradient to ``page`` is the fake score's page less the real one's, ``level * (real
    - fake)``, divided by the mean distance of ``page`` from the real one's (DMD's weighting: the same pull at every
    level) and by the page's size: a descent step moves the page from the fake score's page toward the real one's.
    Neither velocity takes a gradient."""
    with torch.no_grad():
        real_page = noised - level * real
        distance = (page - real_page).abs().mean().clamp_min(torch.finfo(torch.float32).tiny)
        pull = level * (real - fake) / distance
    return 0.5 * F.mse_loss(page, page.detach() - pull)


@contextmanager
def teacher_alone(dit: MossTransformer) -> Generator[MossTransformer]:
    """``dit``, a teacher carrying the fake score's LoRA, as the teacher itself (its LoRA off) for the block."""
    dit.disable_adapters()
    try:
        yield dit
    finally:
        dit.enable_adapters()


def distribution_matching_loss(
    page: Tensor,
    level: float,
    noise: Tensor,
    scores: MossTransformer,
    text: Tensor,
    negative: Tensor,
    settings: DistillSettings,
    references: Sequence[Tensor] | None = None,
) -> Tensor:
    """:func:`matching_loss` of a student's ``page`` noised to ``level`` with ``noise``. ``scores`` is the teacher's
    DiT carrying the fake score's LoRA: with the LoRA off it is the real score, guided at ``settings.guidance_scale``
    against ``negative`` as the teacher teaches (its unconditional pass skipping ``settings.uncond_skip_layers``; at
    scale 1 its conditional pass alone), with it on the fake score, its conditional pass. Both read the prompt and the
    sample page as the student does; gradients reach ``page`` alone."""
    with torch.no_grad():
        noised = (1 - level) * page + level * noise.float()
        with teacher_alone(scores):
            real = _velocity(scores, noised, level, text, references)
            if settings.guidance_scale != 1.0:
                skipped = frozenset(settings.uncond_skip_layers)
                unconditional = _velocity(scores, noised, level, negative, references, skipped)
                real = unconditional + settings.guidance_scale * (real - unconditional)
        fake = _velocity(scores, noised, level, text, references)
    return matching_loss(page, noised, level, real, fake)


def fake_score_loss(
    fake: MossTransformer,
    page: Tensor,
    level: float,
    noise: Tensor,
    text: Tensor,
    references: Sequence[Tensor] | None = None,
) -> Tensor:
    """The fake score's flow-matching loss on a student's ``page`` (held fixed): the page noised to ``level`` with
    ``noise``, and the mean squared error of ``fake``'s velocity there, from the page's prompt and sample page, to
    ``noise - page``. Trained on it, the fake score is the velocity field of the student's own pages."""
    page, noise = page.detach().float(), noise.float()
    noised = (1 - level) * page + level * noise
    return F.mse_loss(_velocity(fake, noised, level, text, references), noise - page)


class DistillRun(BaseModel):
    """A distillation run: its length, the student LoRA's shape (``targets``: the modules it adapts, a regex matched
    in full against the DiT's module names) and learning rate (warmed up over ``warmup`` steps),
    the samples whose losses one optimizer step averages, the share of them drawn from another page's sample page
    (``swap``: a page's layout and text with a look that is not its own, as a new page is drawn), how the student is
    taught (``distill``), the distribution-matching term added to that (``matching``; None: none, and the run draws,
    trains and records its settings as a run did before there was one), and when it is saved; ``stop_at`` ends this
    run at that step with a checkpoint (a chain of short jobs, each resumed from the last)."""

    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=1000, ge=1)
    rank: int = Field(default=64, ge=1)
    alpha: int = Field(default=64, ge=1)
    targets: str = DISTILL_TARGETS
    learning_rate: float = 2e-5
    warmup: int = Field(default=50, ge=1)
    samples_per_step: int = Field(default=1, ge=1)
    swap: float = Field(default=0.0, ge=0.0, le=1.0)
    grad_clip: float = 1.0
    distill: DistillSettings = DistillSettings()
    matching: MatchingSettings | None = Field(default=None, exclude_if=lambda matching: matching is None)
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
    clipping, each sample's pair and the student step it was taught at, and its time; in a run that matches
    distributions, its samples' mean matching loss (unweighted; not in ``loss``) and the fake score's mean
    flow-matching loss over its updates of the step."""

    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    pairs: list[str]
    indices: list[int]
    matching: float | None = Field(default=None, exclude_if=lambda loss: loss is None)
    fake_loss: float | None = Field(default=None, exclude_if=lambda loss: loss is None)
    seconds: float


class _StudentPage(BaseModel):
    """A student's page of a step (held fixed), what it was drawn from, and the noise level and noise of each of the
    fake score's updates on it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    page: Tensor
    text: Tensor
    references: list[Tensor] | None
    draws: list[tuple[float, Tensor]]


def _sum_gradients(parameters: Sequence[torch.nn.Parameter], ranks: Ranks) -> None:
    """Over several ranks, each of ``parameters``' gradients summed over them (a rank without one giving zeros)."""
    if ranks.mesh is None:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad)


def _train_fake_score(
    fake: LoraTraining, pages: Sequence[_StudentPage], updates: int, run: DistillRun, ranks: Ranks
) -> float:
    """The fake score's ``updates`` optimizer steps for one of the student's, each on the mean
    :func:`fake_score_loss` of the step's ``pages`` (this rank's; the ranks sum their gradients) at that update's
    level and noise.

    Returns:
        The mean loss over the updates and the step's samples.
    """
    losses = torch.zeros(updates, dtype=torch.float64, device=run.device)
    for update in range(updates):
        for drawn in pages:
            level, noise = drawn.draws[update]
            loss = fake_score_loss(fake.transformer, drawn.page, level, noise, drawn.text, drawn.references)
            (loss / run.samples_per_step).backward()
            losses[update] += loss.detach().double() / run.samples_per_step
        _sum_gradients(fake.parameters, ranks)
        torch.nn.utils.clip_grad_norm_(fake.parameters, run.grad_clip)
        fake.step()
    if ranks.mesh is not None:
        dist.all_reduce(losses)
    return losses.mean().item()


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

    With ``run.matching`` the teacher's DiT also carries the fake score, a second LoRA with its own AdamW (no third
    copy of the DiT: the teacher's passes run with it off). Each sample's loss takes the matching term of its page
    (:func:`distribution_matching_loss`), and after the student's step the fake score takes its updates on the step's
    pages (:func:`_train_fake_score`). Their levels and noise are drawn from the same generator after the sample's own,
    by such a run alone, and a checkpoint keeps the fake score's LoRA and optimizer state in its ``FAKE_SCORE``
    folder, which a resumed run goes on from.

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
    matching = run.matching
    fake, real = None, run.distill
    if matching is not None:
        shape = LoraSettings(rank=matching.rank, alpha=matching.rank, target_modules=run.targets)
        rate = settings.model_copy(update={"learning_rate": matching.learning_rate})
        fake = LoraTraining(teacher, rate, sample_size_of(cache, None), lora=shape, transformer=frozen)
        if matching.guidance_scale is not None:
            real = run.distill.model_copy(update={"guidance_scale": matching.guidance_scale})
    negative_embeds = load_file(str(negative), device=device)["prompt_embeds"]
    record = DistillRecord(
        teacher=str(teacher.resolve()), student_steps=run.distill.student_steps, shift=run.distill.shift
    )
    generator = torch.Generator().manual_seed(run.seed)
    done = 0
    if resume is not None:
        done = _resumed(training, generator, resume, run, device)
        if fake is not None:
            _restore(fake, resume / FAKE_SCORE, device)
        if ranks.main:
            lines = (out / LOG).read_text(encoding="utf-8").splitlines(keepends=True)
            kept = [line for line in lines if DistillLog.model_validate_json(line).step <= done]
            (out / LOG).write_text("".join(kept), encoding="utf-8")
    if ranks.main:
        out.mkdir(parents=True, exist_ok=True)
    checkpoint = out if resume is None else resume
    last = run.steps if run.stop_at is None else min(run.stop_at, run.steps)
    sampled = [
        entry for entry in entries if entry.reference_size is not None
    ]  # the pages a sample can be borrowed from
    for step in range(done + 1, last + 1):
        started = time.monotonic()
        losses = torch.zeros(run.samples_per_step, dtype=torch.float64, device=device)
        matched = torch.zeros(run.samples_per_step, dtype=torch.float64, device=device)
        pages: list[_StudentPage] = []
        pairs, indices = [], []
        for place in range(run.samples_per_step):
            entry = entries[int(torch.randint(len(entries), (1,), generator=generator))]
            index = int(torch.randint(run.distill.student_steps, (1,), generator=generator))
            noise = torch.randn(target_shape(cache, entry), generator=generator)
            donor = None
            if (
                run.swap > 0 and entry.reference_size is not None
            ):  # drawn only by a run that swaps: others go on as before
                swapped = float(torch.rand((), generator=generator)) < run.swap
                other = sampled[int(torch.randint(len(sampled), (1,), generator=generator))]
                donor = other if swapped and other.pair != entry.pair else None
            pairs.append(entry.pair if donor is None else f"{entry.pair}<-{donor.pair}")
            indices.append(index)
            draws: list[tuple[float, Tensor]] = []
            if matching is not None:  # drawn only by a run that matches: the page's own, then each fake update's
                low, high = matching.levels
                for _ in range(matching.updates + 1):
                    level = low + (high - low) * float(torch.rand((), generator=generator))
                    draws.append((level, torch.randn(noise.shape, generator=generator)))
            if not ranks.owns(place):  # drawn all the same, so every rank's generator goes on alike
                continue
            sample = load_sample(cache, entry, device)
            reference = sample.reference if donor is None else load_sample(cache, donor, device).reference
            references = None if reference is None else [reference]
            with nullcontext() if fake is None else teacher_alone(frozen):
                loss, page = _taught(
                    training.transformer,
                    frozen,
                    noise.to(device),
                    index,
                    sample.prompt_embeds,
                    negative_embeds,
                    run.distill,
                    references,
                )
            losses[place] = loss.detach().double()
            if matching is not None:
                level, fresh = draws[0]
                term = distribution_matching_loss(
                    page, level, fresh.to(device), frozen, sample.prompt_embeds, negative_embeds, real, references
                )
                matched[place] = term.detach().double()
                loss = loss + matching.weight * term
                later = [(level, fresh.to(device)) for level, fresh in draws[1:]]
                pages.append(
                    _StudentPage(page=page.detach(), text=sample.prompt_embeds, references=references, draws=later)
                )
            (loss / run.samples_per_step).backward()
        if ranks.mesh is not None:  # each rank's share of the mean's gradient, summed
            dist.all_reduce(losses)
            if matching is not None:
                dist.all_reduce(matched)
        _sum_gradients(training.parameters, ranks)
        grad_norm = torch.nn.utils.clip_grad_norm_(training.parameters, run.grad_clip).item()
        learning_rate = training.step()
        fake_loss = None
        if matching is not None and fake is not None:
            fake_loss = _train_fake_score(fake, pages, matching.updates, run, ranks)
        if not ranks.main:
            continue
        logged = DistillLog(
            step=step,
            loss=losses.mean().item(),
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            pairs=pairs,
            indices=indices,
            matching=None if matching is None else matched.mean().item(),
            fake_loss=fake_loss,
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
            if fake is not None:
                (checkpoint / FAKE_SCORE).mkdir(exist_ok=True)
                fake.save(checkpoint / FAKE_SCORE, progress)
                torch.save(fake.state_dict(), checkpoint / FAKE_SCORE / OPTIMIZER_STATE)
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
    _restore(training, checkpoint, device)
    generator.set_state(torch.frombuffer(bytearray(progress.generator), dtype=torch.uint8))
    return progress.step


def _restore(training: LoraTraining, folder: Path, device: str) -> None:
    """``training``'s LoRA and optimizer set to those saved in ``folder``.

    Raises:
        ValueError: If the folder holds no LoRA and optimizer state, or its weights are not its LoRA's.
    """
    if not (folder / ADAPTER_WEIGHTS).is_file() or not (folder / OPTIMIZER_STATE).is_file():
        raise ValueError(f"{folder} holds no LoRA and optimizer state to resume from")
    weights = load_file(str(folder / ADAPTER_WEIGHTS), device=device)
    loaded = set_peft_model_state_dict(training.transformer, weights)
    if loaded.unexpected_keys:
        raise ValueError(f"{folder}: weights that are not its LoRA's")
    training.load_state_dict(torch.load(folder / OPTIMIZER_STATE, map_location=device, weights_only=True))


def student_of(teacher: Path, checkpoint: Path, *, device: str = "cuda") -> MossTransformer:
    """The student saved in ``checkpoint``, for drawing: the DiT at ``teacher`` in bfloat16 on ``device`` with the
    checkpoint's LoRA beside its weights as it was trained (each adapted layer adds its LoRA's output to its own),
    the LoRA's weights cast to bfloat16 (:func:`~diffu_moss.train.loop.cast_lora`), in eval mode. Drawn in its
    record's (``DISTILL_RECORD``) steps at guidance 1.

    Raises:
        ValueError: If the checkpoint's weights are not its LoRA's.
    """
    student = MossTransformer.from_pretrained(str(teacher), torch_dtype=torch.bfloat16)
    add_lora(student, AdapterRecord.read(checkpoint / ADAPTER_RECORD).lora)
    loaded = set_peft_model_state_dict(student, load_file(str(checkpoint / ADAPTER_WEIGHTS)))
    if loaded.unexpected_keys:
        raise ValueError(f"{checkpoint}: weights that are not its LoRA's")
    cast_lora(student)
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
    swap: Annotated[
        float,
        typer.Option(min=0.0, max=1.0, help="the share of samples drawn from another page's sample page"),
    ] = 0.0,
    matching: Annotated[
        float,
        typer.Option(
            min=0.0,
            help="the weight of a distribution-matching term (DMD) beside the regression: the student's pages pulled "
            "from where a fake score (a second LoRA on the teacher, trained on them) places them toward where the "
            "teacher does (0: none, as before)",
        ),
    ] = 0.0,
    matching_guidance: Annotated[
        float | None,
        typer.Option(
            min=1.0,
            help="with --matching: the teacher's guidance scale as the real score (default: --guidance-scale; 1: its "
            "conditional pass alone)",
        ),
    ] = None,
    fake_rank: Annotated[
        int | None, typer.Option(min=1, help="with --matching: the fake score's LoRA rank (default: --rank)")
    ] = None,
    fake_learning_rate: Annotated[
        float | None, typer.Option(help="with --matching: the fake score's AdamW's (default: --learning-rate)")
    ] = None,
    fake_updates: Annotated[
        int | None,
        typer.Option(min=1, help="with --matching: the fake score's optimizer steps for each of the student's (5)"),
    ] = None,
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
    fake_options = (matching_guidance, fake_rank, fake_learning_rate, fake_updates)
    if matching == 0 and any(option is not None for option in fake_options):
        raise typer.BadParameter("--matching-guidance and the --fake-* options set the matching term: give --matching")
    matched = None
    if matching > 0:
        given = {
            "weight": matching,
            "guidance_scale": matching_guidance,
            "rank": rank if fake_rank is None else fake_rank,
            "learning_rate": learning_rate if fake_learning_rate is None else fake_learning_rate,
            **({} if fake_updates is None else {"updates": fake_updates}),
        }
        try:  # a ValidationError (a learning rate not above 0) is a ValueError too
            matched = MatchingSettings.model_validate(given)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
    run = DistillRun(
        steps=steps,
        rank=rank,
        alpha=rank,
        learning_rate=learning_rate,
        samples_per_step=samples_per_step,
        swap=swap,
        distill=DistillSettings(
            student_steps=student_steps,
            teacher_steps=teacher_steps,
            guidance_scale=guidance_scale,
            repulsion=repulsion,
        ),
        matching=matched,
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
