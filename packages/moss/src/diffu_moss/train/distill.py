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
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as F

from diffu_moss.modular_pipelines.moss.before_denoise import SAMPLING_SHIFT, sampling_sigmas
from diffu_moss.modular_pipelines.moss.denoise import DEMO_UNCOND_SKIP_LAYERS, GUIDANCE_SCALE
from diffu_moss.transformer import MossTransformer


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
