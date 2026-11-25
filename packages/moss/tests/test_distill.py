"""On-policy distillation on CPU (conftest's tiny MOSS and tiny style pairs): the student's own rollout, the
teacher's guided target across one student step, a loss that is zero for a student that already steps as its teacher,
a LoRA student whose few-step drawing moves toward its teacher's many-step guided drawing when trained on it, and
a distillation run whose checkpoint merges into the teacher's DiT as a student that steps closer to the teacher's
guided steps (a used run directory refused); a run stopped and resumed goes on exactly as the run that went
through, and a run on two processes (torchrun, gloo) trains as one process does up to the rounding of the summed
gradients. The distribution-matching term: no pull where the fake score is the real one and DMD's pull where not, a
toy student moved to its teacher's distribution, a run without it as before it existed, its weight in the student's
gradients, the teacher teaching with the fake score's LoRA off, and a matching run resumed and on two processes as
the run without them. Whole pages matched: the page the student draws with gradients through its steps, the
velocities' difference as the pull, the level's distribution, the new settings left out of an older run's record, a
whole-page run resumed as the run that went through, and a run begun at another run's student."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import torch
from peft import set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import Tensor

from diffu_moss.modular_pipelines.moss.before_denoise import sampling_sigmas
from diffu_moss.train.cache import CachedSample, load_sample, read_entries, target_shape
from diffu_moss.train.distill import (
    DISTILL_RECORD,
    DISTILL_TARGETS,
    FAKE_SCORE,
    DistillLog,
    DistillRun,
    DistillSettings,
    MatchingPull,
    MatchingSettings,
    distill,
    distillation_loss,
    distribution_matching_loss,
    drawn,
    latest_checkpoint,
    matching_loss,
    rolled_out,
    student_levels,
    student_of,
    teacher_alone,
    teacher_levels,
    teacher_velocities,
    teacher_velocity,
)
from diffu_moss.train.loop import LOG, TrainSettings, lora_transformer
from diffu_moss.train.records import PROGRESS_RECORD, LoraSettings, Progress
from diffu_moss.transformer import MossTransformer

SETTINGS = DistillSettings(student_steps=2, teacher_steps=8, guidance_scale=3.0, uncond_skip_layers=(1,))


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


def test_the_teachers_velocity_takes_a_state_where_its_guided_steps_through_its_own_levels_arrive(
    teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    state, text = _noise(sample), sample.prompt_embeds
    levels = student_levels(SETTINGS)
    start, end = levels[0], levels[1]
    between = teacher_levels(start, end, SETTINGS)
    eight = [*sampling_sigmas(8, SETTINGS.shift), 0.0]

    velocity = teacher_velocity(teacher, state, start, end, text, negative, SETTINGS, [_page(sample)])

    arrived = state.clone()
    assert between == [level for level in eight if end <= level <= start] and len(between) > 2
    for now, after in pairwise(between):
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


def test_the_loss_is_zero_for_a_student_that_steps_as_its_unguided_teacher_on_the_same_levels(
    teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    same = DistillSettings(student_steps=2, teacher_steps=2, guidance_scale=1.0)

    losses = [
        distillation_loss(teacher, teacher, _noise(sample), index, sample.prompt_embeds, negative, same).item()
        for index in (0, 1)
    ]
    guided = distillation_loss(teacher, teacher, _noise(sample), 0, sample.prompt_embeds, negative, SETTINGS).item()

    assert losses == pytest.approx([0.0, 0.0], abs=1e-10)
    assert guided > 1e-6  # a guided, finer teacher asks for another step than the student's own
    with pytest.raises(ValueError, match="not one of the student's 2"):
        distillation_loss(teacher, teacher, _noise(sample), 2, sample.prompt_embeds, negative, same)


def test_repulsion_pushes_the_student_from_the_teachers_unguided_velocity_across_the_step(
    teacher: MossTransformer, sample: CachedSample, negative: Tensor
) -> None:
    state, text, references = _noise(sample), sample.prompt_embeds, [_page(sample)]
    levels = student_levels(SETTINGS)
    start, end = levels[0], levels[1]
    pushed = SETTINGS.model_copy(update={"repulsion": 0.25})

    guided, unguided = teacher_velocities(teacher, state, start, end, text, negative, SETTINGS, references)
    loss = distillation_loss(teacher, teacher, state, 0, text, negative, pushed, references).item()

    assert unguided is not None
    arrived, summed = state.clone(), torch.zeros_like(state)
    for now, after in pairwise(teacher_levels(start, end, SETTINGS)):  # the unconditional passes along the guided path
        conditional = _step(teacher, arrived, now, text, sample)
        with torch.no_grad():
            unconditional = teacher(
                arrived, torch.tensor([now * 1000]), negative, None, frozenset({1}), references=references
            ).float()
        summed = summed + (after - now) * unconditional
        arrived = arrived + (after - now) * (unconditional + 3.0 * (conditional - unconditional))
    assert torch.allclose(unguided, summed / (end - start), atol=1e-5)
    own = _step(teacher, state, start, text, sample)
    plain = (own - guided).square().mean().item()
    assert loss == pytest.approx(plain - 0.25 * (own - unguided).square().mean().item(), rel=1e-4)
    assert loss != pytest.approx(plain, rel=1e-3)
    least = guided + 0.25 / 0.75 * (guided - unguided)  # where the pushed loss is least: its gradient is zero there
    velocity = least.clone().requires_grad_(True)
    ((velocity - guided).square().mean() - 0.25 * (velocity - unguided).square().mean()).backward()
    assert velocity.grad is not None and velocity.grad.abs().max().item() < 1e-9
    alone = SETTINGS.model_copy(update={"guidance_scale": 1.0})
    assert teacher_velocities(teacher, state, start, end, text, negative, alone, references)[1] is None


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


def test_a_distillation_run_saves_a_lora_over_the_noise_levels_layers_too_whose_student_steps_closer_to_its_teacher(
    tiny_moss_dit: Path, tiny_pair_cache: Path, teacher: MossTransformer, negative: Tensor, tmp_path: Path
) -> None:
    save_file({"prompt_embeds": negative}, str(tmp_path / "negative.safetensors"))
    run = DistillRun(
        steps=40, rank=8, alpha=8, learning_rate=3e-3, warmup=2, distill=SETTINGS, save_every=40, device="cpu"
    )

    checkpoint = distill(tiny_moss_dit, tiny_pair_cache, tmp_path / "negative.safetensors", tmp_path / "run", run)

    logged = [json.loads(line) for line in (tmp_path / "run" / LOG).read_text(encoding="utf-8").splitlines()]
    assert [row["step"] for row in logged] == list(range(1, 41))
    assert {index for row in logged for index in row["indices"]} == {0, 1}  # both of the student's steps taught
    record = json.loads((checkpoint / DISTILL_RECORD).read_text(encoding="utf-8"))
    assert (record["student_steps"], record["guidance_scale"]) == (2, 1.0)
    adapted = {key.split(".lora_")[0] for key in load_file(str(checkpoint / "adapter.safetensors"))}
    assert {"time_embedding.0", "time_embedding.2", "shared_block_modulation", "video_head_modulation"} <= adapted
    assert {"video_embedder", "final_linear_video", "layers.0.proj.q"} <= adapted
    student = student_of(tiny_moss_dit, checkpoint, device="cpu")
    lora = {name: parameter.dtype for name, parameter in student.named_parameters() if ".lora_" in name}
    assert lora and set(lora.values()) == {torch.bfloat16}  # beside the DiT's weights, in their dtype: not merged
    student = student.float()
    entries = read_entries(tiny_pair_cache)

    def loss_of(model: MossTransformer) -> float:  # on noise the run never drew, at both steps of every entry
        total = 0.0
        for place, entry in enumerate(entries):
            held = load_sample(tiny_pair_cache, entry, "cpu")
            references = None if held.reference is None else [held.reference]
            for index in (0, 1):
                noise = _noise(held, 1000 + place)
                with torch.no_grad():
                    total += distillation_loss(
                        model, teacher, noise, index, held.prompt_embeds, negative, SETTINGS, references
                    ).item()
        return total

    assert loss_of(student) < 0.7 * loss_of(teacher)
    with pytest.raises(ValueError, match="holds a training already"):
        distill(tiny_moss_dit, tiny_pair_cache, tmp_path / "negative.safetensors", tmp_path / "run", run)


def test_a_run_that_swaps_teaches_pages_from_other_pages_samples(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    save_file({"prompt_embeds": negative}, str(tmp_path / "negative.safetensors"))
    sampled = {entry.pair for entry in read_entries(tiny_pair_cache) if entry.reference_size is not None}
    assert len(sampled) > 1  # the fixture has pages to borrow a sample from
    run = DistillRun(steps=3, rank=4, alpha=4, warmup=1, distill=SETTINGS, samples_per_step=4, swap=1.0, device="cpu")

    distill(tiny_moss_dit, tiny_pair_cache, tmp_path / "negative.safetensors", tmp_path / "run", run)

    lines = (tmp_path / "run" / LOG).read_text(encoding="utf-8").splitlines()
    taught = [str(pair) for line in lines for pair in json.loads(line)["pairs"]]
    swaps = [pair.split("<-") for pair in taught if "<-" in pair]
    assert swaps and all(page != donor and page in sampled and donor in sampled for page, donor in swaps)


def _logged(run: Path) -> list[DistillLog]:
    return [DistillLog.model_validate_json(line) for line in (run / LOG).read_text(encoding="utf-8").splitlines()]


def _rows(run: Path) -> list[dict[str, object]]:
    lines = (run / LOG).read_text(encoding="utf-8").splitlines()
    return [{key: value for key, value in json.loads(line).items() if key != "seconds"} for line in lines]


def test_a_distillation_stopped_and_resumed_goes_on_exactly_as_the_run_that_went_through(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(
        steps=4, rank=4, alpha=4, learning_rate=3e-3, warmup=2, distill=SETTINGS, save_every=4, device="cpu"
    )
    through = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "through", run)

    stopped = run.model_copy(update={"stop_at": 2})
    distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", stopped)
    first = latest_checkpoint(tmp_path / "chained")
    resumed = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", run, resume=first)

    assert first is not None and first.name == "step_000002" and resumed.name == "step_000004"
    assert _rows(tmp_path / "chained") == _rows(tmp_path / "through")
    weights = [load_file(str(folder / "adapter.safetensors")) for folder in (resumed, through)]
    assert [name for name in weights[1] if not torch.equal(weights[0][name], weights[1][name])] == []
    other = run.model_copy(update={"learning_rate": 1e-3})
    with pytest.raises(ValueError, match="other settings"):
        distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", other, resume=first)


def test_a_distillation_on_two_processes_trains_as_one_process_does(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))

    def launched(world: int) -> Path:
        out = tmp_path / f"world_{world}"
        command = [
            *(sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"),
            *("-m", "diffu_moss.train.distill", "--device", "cpu", "--teacher", str(tiny_moss_dit)),
            *("--cache", str(tiny_pair_cache), "--negative", str(file), "--out", str(out)),
            *("--steps", "3", "--samples-per-step", "2", "--rank", "4", "--learning-rate", "3e-3"),
            *("--student-steps", "2", "--teacher-steps", "8", "--save-every", "3"),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-3000:]
        return out

    one, two = launched(1), launched(2)

    rows = _rows(one), _rows(two)
    assert [(row["step"], row["pairs"], row["indices"]) for row in rows[0]] == [
        (row["step"], row["pairs"], row["indices"]) for row in rows[1]
    ]
    for ours, theirs in zip(*rows, strict=True):
        assert ours["loss"] == pytest.approx(theirs["loss"], rel=1e-4, abs=1e-8)
    weights = [load_file(str(out / "checkpoints" / "step_000003" / "adapter.safetensors")) for out in (one, two)]
    far = [name for name in weights[0] if not torch.allclose(weights[0][name], weights[1][name], rtol=1e-3, atol=1e-6)]
    assert far == []


MATCHING = MatchingSettings(rank=4, learning_rate=3e-3, updates=2)


def _with_fake_score(tiny_moss_dit: Path) -> MossTransformer:
    """The tiny teacher in float32 carrying a fresh fake-score LoRA (its B zero: the teacher itself)."""
    lora = LoraSettings(rank=4, alpha=4, target_modules=DISTILL_TARGETS)
    scores = lora_transformer(tiny_moss_dit, lora, device="cpu", seed=0).float().eval()
    scores.disable_gradient_checkpointing()
    return scores


def test_the_matching_gradient_is_zero_where_the_fake_score_is_the_real_one_and_else_dmds_pull(
    tiny_moss_dit: Path, sample: CachedSample, negative: Tensor
) -> None:
    scores = _with_fake_score(tiny_moss_dit)
    conditional = SETTINGS.model_copy(update={"guidance_scale": 1.0})
    text, references, level, noise = sample.prompt_embeds, [_page(sample)], 0.6, _noise(sample, 7)

    def gradient() -> Tensor:
        page = _noise(sample, 3).requires_grad_(True)
        distribution_matching_loss(page, level, noise, scores, text, negative, conditional, references).backward()
        assert page.grad is not None
        return page.grad

    equal = gradient()
    generator = torch.Generator().manual_seed(5)
    for name, parameter in scores.named_parameters():
        if ".lora_B" in name:
            parameter.data = 0.05 * torch.randn(parameter.shape, generator=generator)
    moved = gradient()

    assert torch.count_nonzero(equal).item() == 0
    page = _noise(sample, 3)
    noised = (1 - level) * page + level * noise
    with teacher_alone(scores):
        real = _step(scores, noised, level, text, sample)
    fake = _step(scores, noised, level, text, sample)
    distance = (page - (noised - level * real)).abs().mean()
    assert torch.allclose(moved, level * (real - fake) / distance / page.numel(), rtol=1e-4, atol=1e-12)
    assert moved.abs().max().item() > 0


def test_matching_moves_a_students_pages_to_the_teachers_distribution() -> None:
    """A toy: the teacher's pages are Gaussian about ``target``, the student's about its learnt ``mean``, each score the
    exact velocity of its distribution; the student's mean, trained on the matching loss alone, reaches the target."""
    spread, size = 0.5, 64
    target = torch.linspace(-1.5, 1.5, size)
    mean = torch.zeros(size, requires_grad=True)
    generator = torch.Generator().manual_seed(0)

    def velocity(
        noised: Tensor, level: float, centre: Tensor
    ) -> Tensor:  # E[noise - page | noised] for N(centre, spread²)
        expected = centre + (1 - level) * spread**2 / ((1 - level) ** 2 * spread**2 + level**2) * (
            noised - (1 - level) * centre
        )
        return (noised - expected) / level

    optimizer = torch.optim.Adam([mean], lr=0.05)
    for _ in range(300):
        page = mean + spread * torch.randn(size, generator=generator)
        level = 0.02 + 0.96 * torch.rand((), generator=generator).item()
        noised = ((1 - level) * page + level * torch.randn(size, generator=generator)).detach()
        loss = matching_loss(
            page, noised, level, velocity(noised, level, target), velocity(noised, level, mean.detach())
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert (mean.detach() - target).abs().max().item() < 0.15


def test_a_run_without_matching_draws_and_trains_as_before_the_option(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(
        steps=3,
        rank=4,
        alpha=4,
        learning_rate=3e-3,
        warmup=2,
        distill=SETTINGS,
        samples_per_step=2,
        swap=0.5,
        save_every=3,
        device="cpu",
    )

    checkpoint = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "run", run)

    before = [  # this run's log as the code before the option wrote it
        (1, 0.016789900371804833, 0.0015, 0.04360968619585037, ["vol_04<-vol_10", "vol_00"], [1, 0]),
        (2, 0.01574396132491529, 0.003, 0.05015214905142784, ["vol_04<-vol_10", "vol_05"], [1, 0]),
        (3, 0.003178988699801266, 0.003, 0.02162192016839981, ["vol_01", "vol_09"], [1, 1]),
    ]
    rows = _rows(tmp_path / "run")
    assert [set(row) for row in rows] == [{"step", "loss", "learning_rate", "grad_norm", "pairs", "indices"}] * 3
    for row, (step, loss, rate, norm, pairs, indices) in zip(rows, before, strict=True):
        assert (row["step"], row["learning_rate"], row["pairs"], row["indices"]) == (step, rate, pairs, indices)
        assert row["loss"] == pytest.approx(loss, rel=1e-5) and row["grad_norm"] == pytest.approx(norm, rel=1e-5)
    weights = load_file(str(checkpoint / "adapter.safetensors"))
    assert sum(tensor.double().abs().sum().item() for tensor in weights.values()) == pytest.approx(
        449.88732852, rel=1e-6
    )
    settings = json.loads(Progress.read(checkpoint / PROGRESS_RECORD).settings)
    assert "matching" not in settings and not (checkpoint / FAKE_SCORE).exists()  # older checkpoints resume


def test_the_matching_terms_weight_reaches_the_students_gradients(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(steps=1, rank=4, alpha=4, learning_rate=3e-3, warmup=2, distill=SETTINGS, device="cpu")

    for weight in (1e-3, 30.0):
        matching = MATCHING.model_copy(update={"weight": weight})
        distill(
            tiny_moss_dit, tiny_pair_cache, file, tmp_path / str(weight), run.model_copy(update={"matching": matching})
        )

    light, heavy = (_logged(tmp_path / str(weight))[0] for weight in (1e-3, 30.0))
    assert (light.pairs, light.indices) == (heavy.pairs, heavy.indices)
    assert light.loss == pytest.approx(heavy.loss, rel=1e-6)  # the regression alone is logged as loss
    assert light.matching is not None and light.matching > 0
    assert light.matching == pytest.approx(heavy.matching, rel=1e-6)
    assert heavy.grad_norm > 1.2 * light.grad_norm


def test_the_teacher_teaches_without_the_fake_scores_lora(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(
        steps=2,
        rank=4,
        alpha=4,
        learning_rate=3e-3,
        warmup=1,
        distill=SETTINGS,
        matching=MATCHING,
        save_every=1,
        device="cpu",
    )

    distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "run", run)

    first = tmp_path / "run" / "checkpoints" / "step_000001"
    assert load_file(str(first / FAKE_SCORE / "adapter.safetensors"))  # the fake score had trained by step 2
    lora = LoraSettings(rank=4, alpha=4, target_modules=DISTILL_TARGETS)
    student = lora_transformer(tiny_moss_dit, lora, device="cpu", seed=0).eval()
    set_peft_model_state_dict(student, load_file(str(first / "adapter.safetensors")))
    plain = MossTransformer.from_pretrained(str(tiny_moss_dit), torch_dtype=torch.bfloat16).eval()
    generator = torch.Generator()
    generator.set_state(
        torch.frombuffer(bytearray(Progress.read(first / PROGRESS_RECORD).generator), dtype=torch.uint8)
    )
    entries = read_entries(tiny_pair_cache)
    entry = entries[int(torch.randint(len(entries), (1,), generator=generator))]
    index = int(torch.randint(SETTINGS.student_steps, (1,), generator=generator))
    noise = torch.randn(target_shape(tiny_pair_cache, entry), generator=generator)
    held = load_sample(tiny_pair_cache, entry, "cpu")
    references = None if held.reference is None else [held.reference]
    with torch.no_grad():
        taught = distillation_loss(student, plain, noise, index, held.prompt_embeds, negative, SETTINGS, references)
    row = _rows(tmp_path / "run")[1]
    assert (row["pairs"], row["indices"]) == ([entry.pair], [index])
    assert row["loss"] == pytest.approx(taught.item(), rel=1e-4)


def test_a_matching_run_stopped_and_resumed_goes_on_exactly_as_the_run_that_went_through(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(
        steps=4,
        rank=4,
        alpha=4,
        learning_rate=3e-3,
        warmup=2,
        distill=SETTINGS,
        matching=MATCHING,
        save_every=4,
        device="cpu",
    )
    through = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "through", run)

    distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", run.model_copy(update={"stop_at": 2}))
    first = latest_checkpoint(tmp_path / "chained")
    resumed = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", run, resume=first)

    assert first is not None and resumed.name == "step_000004"
    rows = _rows(tmp_path / "chained")
    assert rows == _rows(tmp_path / "through")
    logged = _logged(tmp_path / "chained")
    assert all((row.fake_loss or 0) > 0 and (row.matching or 0) > 0 for row in logged)
    for saved in ("adapter.safetensors", f"{FAKE_SCORE}/adapter.safetensors"):
        weights = [load_file(str(folder / saved)) for folder in (resumed, through)]
        assert [name for name in weights[1] if not torch.equal(weights[0][name], weights[1][name])] == []
    plain = run.model_copy(update={"matching": None})
    with pytest.raises(ValueError, match="other settings"):
        distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", plain, resume=first)


def test_a_matching_distillation_on_two_processes_trains_as_one_process_does(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))

    def launched(world: int) -> Path:
        out = tmp_path / f"world_{world}"
        command = [
            *(sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"),
            *("-m", "diffu_moss.train.distill", "--device", "cpu", "--teacher", str(tiny_moss_dit)),
            *("--cache", str(tiny_pair_cache), "--negative", str(file), "--out", str(out)),
            *("--steps", "2", "--samples-per-step", "2", "--rank", "4", "--learning-rate", "3e-3"),
            *("--student-steps", "2", "--teacher-steps", "8", "--save-every", "2"),
            *("--matching", "1.0", "--fake-updates", "2"),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        assert done.returncode == 0, done.stderr[-3000:]
        return out

    one, two = launched(1), launched(2)

    rows = _rows(one), _rows(two)
    assert [(row["step"], row["pairs"], row["indices"]) for row in rows[0]] == [
        (row["step"], row["pairs"], row["indices"]) for row in rows[1]
    ]
    for ours, theirs in zip(*rows, strict=True):
        for key in ("loss", "matching", "fake_loss"):
            assert ours[key] == pytest.approx(theirs[key], rel=1e-4, abs=1e-8)
    for saved in ("adapter.safetensors", f"{FAKE_SCORE}/adapter.safetensors"):
        weights = [load_file(str(out / "checkpoints" / "step_000002" / saved)) for out in (one, two)]
        far = [
            name for name in weights[0] if not torch.allclose(weights[0][name], weights[1][name], rtol=1e-3, atol=1e-6)
        ]
        assert far == []


def test_the_student_draws_its_page_through_all_its_levels_with_gradients_through_every_step(
    tiny_moss_dit: Path, sample: CachedSample
) -> None:
    student = _with_fake_score(tiny_moss_dit)  # a LoRA to take the gradients
    generator = torch.Generator().manual_seed(5)
    for name, parameter in student.named_parameters():
        if ".lora_B" in name:
            parameter.data = 0.05 * torch.randn(parameter.shape, generator=generator)
    levels, noise, text = student_levels(SETTINGS), _noise(sample), sample.prompt_embeds

    page = drawn(student, noise, levels, text, [_page(sample)])

    first = noise + (levels[1] - levels[0]) * _step(student, noise, levels[0], text, sample)
    assert torch.allclose(
        page, first + (levels[2] - levels[1]) * _step(student, first, levels[1], text, sample), atol=1e-5
    )
    weights = [parameter for name, parameter in student.named_parameters() if ".lora_B" in name]
    whole = torch.autograd.grad(page.square().mean(), weights, allow_unused=True)
    last = drawn(student, first, levels[1:], text, [_page(sample)])  # the last step alone, from the same state
    alone = torch.autograd.grad(last.square().mean(), weights, allow_unused=True)
    pairs = [(a, b) for a, b in zip(whole, alone, strict=True) if a is not None and b is not None]
    apart = sum((a - b).square().sum().item() for a, b in pairs) ** 0.5
    assert apart > 0.01 * sum(b.square().sum().item() for _, b in pairs) ** 0.5  # the first step's part too


def test_the_velocity_pull_is_the_velocities_difference_at_every_level() -> None:
    generator = torch.Generator().manual_seed(0)
    real, fake, noised = (torch.randn(1, 4, 6, 6, generator=generator) for _ in range(3))

    def gradient(level: float) -> Tensor:
        page = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
        matching_loss(page, noised, level, real, fake, MatchingPull.VELOCITY).backward()
        assert page.grad is not None
        return page.grad

    assert torch.allclose(gradient(0.3), (real - fake) / real.numel(), rtol=1e-5, atol=1e-12)
    assert torch.equal(gradient(0.3), gradient(0.9))


def test_a_pages_matching_level_is_uniform_or_nearer_to_noise_and_the_new_settings_stay_out_of_older_records() -> None:
    uniform, skewed = MatchingSettings(), MatchingSettings(level_beta=2.5)

    assert [uniform.level(share) for share in (0.0, 0.25, 0.5)] == pytest.approx([0.02, 0.26, 0.5])
    assert skewed.level(0.25) == pytest.approx(0.02 + 0.96 * 0.25**0.4) and skewed.level(0.25) > uniform.level(0.25)
    assert set(json.loads(uniform.model_dump_json())) == {
        "weight",
        "guidance_scale",
        "levels",
        "rank",
        "learning_rate",
        "updates",
    }
    assert "betas" not in json.loads(DistillRun().model_dump_json())
    changed = DistillRun(
        betas=(0.0, 0.999), matching=MatchingSettings(whole=True, pull=MatchingPull.VELOCITY, level_beta=2.5)
    )
    record = json.loads(changed.model_dump_json())
    assert record["betas"] == [0.0, 0.999]
    assert (record["matching"]["whole"], record["matching"]["pull"], record["matching"]["level_beta"]) == (
        True,
        "velocity",
        2.5,
    )


def test_a_run_matching_whole_pages_trains_without_the_regression_and_resumes_as_the_run_that_went_through(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    whole = MATCHING.model_copy(update={"whole": True, "pull": MatchingPull.VELOCITY, "level_beta": 2.5})
    run = DistillRun(
        steps=4,
        rank=4,
        alpha=4,
        learning_rate=3e-3,
        warmup=2,
        betas=(0.0, 0.999),
        distill=SETTINGS,
        matching=whole,
        save_every=4,
        device="cpu",
    )
    through = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "through", run)

    distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", run.model_copy(update={"stop_at": 2}))
    first = latest_checkpoint(tmp_path / "chained")
    resumed = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", run, resume=first)

    logged = _logged(tmp_path / "through")
    assert all(row.loss == 0 and (row.matching or 0) > 0 and row.grad_norm > 0 for row in logged)
    assert _rows(tmp_path / "chained") == _rows(tmp_path / "through")
    weights = [load_file(str(folder / "adapter.safetensors")) for folder in (resumed, through)]
    assert [name for name in weights[1] if not torch.equal(weights[0][name], weights[1][name])] == []
    assert any(tensor.abs().max().item() > 0 for name, tensor in weights[1].items() if ".lora_B" in name)
    stepwise = run.model_copy(update={"matching": MATCHING})
    with pytest.raises(ValueError, match="other settings"):
        distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "chained", stepwise, resume=first)


def test_a_run_begins_at_another_runs_student_and_refuses_a_lora_of_another_rank(
    tiny_moss_dit: Path, tiny_pair_cache: Path, negative: Tensor, tmp_path: Path
) -> None:
    file = tmp_path / "negative.safetensors"
    save_file({"prompt_embeds": negative}, str(file))
    run = DistillRun(
        steps=3, rank=4, alpha=4, learning_rate=3e-3, warmup=1, distill=SETTINGS, save_every=3, device="cpu"
    )
    earlier = distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "earlier", run)
    one = run.model_copy(update={"steps": 1, "save_every": 1})

    distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "begun", one, start=earlier)

    begun, fresh = _logged(tmp_path / "begun")[0], _logged(tmp_path / "earlier")[0]
    assert (begun.pairs, begun.indices) == (fresh.pairs, fresh.indices)  # the same sample, another student
    assert begun.loss != pytest.approx(fresh.loss, rel=1e-3)
    wider = one.model_copy(update={"rank": 8, "alpha": 8})
    with pytest.raises(ValueError, match="not its LoRA's"):
        distill(tiny_moss_dit, tiny_pair_cache, file, tmp_path / "wider", wider, start=earlier)


def test_a_page_drawn_with_gradients_through_its_last_step_alone_takes_that_steps_gradient(
    tiny_moss_dit: Path, sample: CachedSample
) -> None:
    student = _with_fake_score(tiny_moss_dit)
    generator = torch.Generator().manual_seed(5)
    for name, parameter in student.named_parameters():
        if ".lora_B" in name:
            parameter.data = 0.05 * torch.randn(parameter.shape, generator=generator)
    levels, noise, text, references = student_levels(SETTINGS), _noise(sample), sample.prompt_embeds, [_page(sample)]
    weights = [parameter for name, parameter in student.named_parameters() if ".lora_B" in name]

    page = drawn(student, noise, levels, text, references, grad_steps=1)

    assert torch.allclose(page, drawn(student, noise, levels, text, references), atol=1e-5)
    state = rolled_out(student, noise, levels, len(levels) - 2, text, references)
    last = drawn(student, state, levels[-2:], text, references)
    ours = torch.autograd.grad(page.square().mean(), weights, allow_unused=True)
    theirs = torch.autograd.grad(last.square().mean(), weights, allow_unused=True)
    for a, b in zip(ours, theirs, strict=True):
        assert (a is None) == (b is None)
        assert a is None or b is None or torch.allclose(a, b, rtol=1e-4, atol=1e-8)
    assert json.loads(MatchingSettings(grad_steps=2).model_dump_json())["grad_steps"] == 2
