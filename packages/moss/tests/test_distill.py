"""On-policy distillation on CPU (conftest's tiny MOSS and tiny style pairs): the student's own rollout, the
teacher's guided target across one student step, a loss that is zero for a student that already steps as its teacher,
and a LoRA student whose few-step drawing moves toward its teacher's many-step guided drawing when trained on it."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import torch
from torch import Tensor

from diffu_moss.train.cache import CachedSample, load_sample, read_entries
from diffu_moss.train.distill import (
    DistillSettings,
    distillation_loss,
    rolled_out,
    student_levels,
    teacher_velocity,
)
from diffu_moss.train.loop import TrainSettings, lora_transformer
from diffu_moss.transformer import MossTransformer

SETTINGS = DistillSettings(student_steps=2, teacher_substeps=3, guidance_scale=3.0, uncond_skip_layers=(1,))


@pytest.fixture(scope="module")
def teacher(tiny_moss_dit: Path) -> MossTransformer:
    return MossTransformer.from_pretrained(str(tiny_moss_dit)).eval()


@pytest.fixture(scope="module")
def sample(tiny_pair_cache: Path) -> CachedSample:
    with_sample = [entry for entry in read_entries(tiny_pair_cache) if entry.reference_size is not None]
    return load_sample(tiny_pair_cache, with_sample[0], "cpu")


@pytest.fixture(scope="module")
def negative(teacher: MossTransformer) -> Tensor:
    return torch.randn(1, 5, teacher.config.text_dim, generator=torch.Generator().manual_seed(1))


def _page(sample: CachedSample) -> Tensor:
    """The sample page's latents (the fixture's entry has one)."""
    assert sample.reference is not None
    return sample.reference


def _noise(sample: CachedSample, seed: int = 0) -> Tensor:
    return torch.randn(sample.target.shape, generator=torch.Generator().manual_seed(seed))


def _step(dit: MossTransformer, latents: Tensor, sigma: float, text: Tensor, sample: CachedSample) -> Tensor:
    with torch.no_grad():
        return dit(latents, torch.tensor([sigma * 1000]), text, None, references=[_page(sample)]).float()


def test_the_student_rolls_out_from_noise_in_unguided_euler_steps_with_its_sample_page(
    teacher: MossTransformer, sample: CachedSample
) -> None:
    levels = student_levels(SETTINGS)
    noise = _noise(sample)

    states = [rolled_out(teacher, noise, levels, index, sample.prompt_embeds, [_page(sample)]) for index in (0, 1)]

    assert levels[0] == pytest.approx(1.0) and levels[-1] == 0.0 and len(levels) == SETTINGS.student_steps + 1
    assert torch.equal(states[0], noise)
    stepped = noise + (levels[1] - levels[0]) * _step(teacher, noise, levels[0], sample.prompt_embeds, sample)
    assert torch.allclose(states[1], stepped, atol=1e-6)
    assert not torch.allclose(states[1], rolled_out(teacher, noise, levels, 1, sample.prompt_embeds, None), atol=1e-4)


def test_the_teachers_velocity_takes_a_state_where_its_guided_sub_steps_arrive(
    teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    state, text = _noise(sample), sample.prompt_embeds
    start, end = 0.9, 0.3

    velocity = teacher_velocity(teacher, state, start, end, text, negative, SETTINGS, [_page(sample)])

    arrived = state.clone()
    for now, after in pairwise(torch.linspace(start, end, SETTINGS.teacher_substeps + 1).tolist()):
        conditional = _step(teacher, arrived, now, text, sample)
        with torch.no_grad():
            unconditional = teacher(
                arrived, torch.tensor([now * 1000]), negative, None, frozenset({1}), references=[_page(sample)]
            ).float()
        arrived = arrived + (after - now) * (unconditional + 3.0 * (conditional - unconditional))
    assert torch.allclose(state + (end - start) * velocity, arrived, atol=1e-5)
    unguided = SETTINGS.model_copy(update={"guidance_scale": 1.0})
    plain = teacher_velocity(teacher, state, start, end, text, negative, unguided, [_page(sample)])
    assert not torch.allclose(plain, velocity, atol=1e-4)  # the guidance moves the target


def test_the_loss_is_zero_for_a_student_that_steps_as_its_unguided_teacher_in_one_sub_step(
    teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    same = DistillSettings(student_steps=2, teacher_substeps=1, guidance_scale=1.0)

    losses = [
        distillation_loss(teacher, teacher, _noise(sample), index, sample.prompt_embeds, negative, same).item()
        for index in (0, 1)
    ]
    guided = distillation_loss(teacher, teacher, _noise(sample), 0, sample.prompt_embeds, negative, SETTINGS).item()

    assert losses == pytest.approx([0.0, 0.0], abs=1e-10)
    assert guided > 1e-6  # a guided, finer teacher asks for another step than the student's own
    with pytest.raises(ValueError, match="not one of the student's 2"):
        distillation_loss(teacher, teacher, _noise(sample), 2, sample.prompt_embeds, negative, same)


def test_a_lora_student_trained_on_the_loss_draws_closer_to_its_teachers_guided_drawing(
    tiny_moss_dit: Path, teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    student = lora_transformer(tiny_moss_dit, TrainSettings(rank=8, alpha=8).lora, device="cpu", seed=0).float()
    student.disable_gradient_checkpointing()
    text, references = sample.prompt_embeds, [_page(sample)]
    levels = student_levels(SETTINGS)

    def gap() -> float:  # the student's 2-step drawing against the teacher's guided drawing through the same levels
        distances = []
        for seed in (100, 101, 102):  # noise the training never drew
            noise = _noise(sample, seed)
            drawn = rolled_out(student, noise, levels, len(levels) - 1, text, references)
            taught = noise.clone()
            for now, after in pairwise(levels):
                taught = taught + (after - now) * teacher_velocity(
                    teacher, taught, now, after, text, negative, SETTINGS, references
                )
            distances.append((drawn - taught).square().mean().item())
        return sum(distances) / len(distances)

    before = gap()
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=3e-3)
    for step in range(60):
        loss = distillation_loss(student, teacher, _noise(sample, step), step % 2, text, negative, SETTINGS, references)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert gap() < 0.5 * before
