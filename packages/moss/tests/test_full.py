"""MOSS-Image's full fine-tune on CPU (conftest's tiny MOSS and tiny style pairs): MOSS's optimizer recipe (Muon on the
joint blocks' matrices, AdamW on the rest, no weight decay on biases, norms and embedders, the last block's q/k norm
weights decayed towards one), a step that trains every weight in float32, a checkpoint that diffu-moss-init draws
with, the average of the weights by its formula, and a run resumed from a checkpoint's state going on exactly as it
would have, and refused one it was not trained with."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file
from torch import Tensor

from diffu.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.train.full import AVERAGE, AVERAGE_WEIGHTS, DIT, STATE, FullOptimizer, FullSettings
from diffu_moss.train.loop import LOG, TrainSettings, train
from diffu_moss.transformer import MossTransformer

JOINT_MATRICES = [
    f"layers.{k}.proj.{name}.weight" for k in range(3) for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")
]
DECAYED = [  # the tiny DiT's other matrices: the text refiner's, the modulations' and the output's
    *(
        f"text_embedder.blocks.{k}.{name}.weight"
        for k in range(2)
        for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")
    ),
    "shared_block_modulation.weight",
    "video_head_modulation.weight",
    "final_linear_video.weight",
]
DECAY = 0.9
# 3 steps of 2 samples, the learning rate rising over 4 steps (so a resumed run must restore its schedule), an average
# of the weights, a checkpoint and its state after every step
FULL_RUN = TrainSettings(steps=3, save_every=1, warmup=4, full=FullSettings(ema=DECAY, save_state=True), device="cpu")


def _weights(folder: Path) -> dict[str, Tensor]:
    """The weights of the tiny DiT folder ``folder`` (one safetensors file), in the dtype they were saved in."""
    return load_file(str(folder / SAFETENSORS_WEIGHTS_NAME))


def _checkpoint(run: Path, step: int) -> Path:
    return run / "checkpoints" / f"step_{step:06d}"


@pytest.fixture(scope="module")
def bfloat16_dit(tiny_moss_dit: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny MOSS's DiT saved in bfloat16, as MOSS-Image's Base is released."""
    folder = tmp_path_factory.mktemp("bfloat16_dit")
    MossTransformer.from_pretrained(str(tiny_moss_dit), torch_dtype=torch.bfloat16).save_pretrained(str(folder))
    return folder


@pytest.fixture(scope="module")
def full_run(bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A full fine-tune (``FULL_RUN``) of the bfloat16 tiny DiT on the tiny pairs' cache, for the tests of what reads
    its checkpoints. Its run directory."""
    run = tmp_path_factory.mktemp("full_run") / "run"
    train(bfloat16_dit, tiny_pair_cache, run, FULL_RUN)
    return run


def test_full_optimizer_steps_the_joint_blocks_matrices_by_muon_and_the_rest_by_adamw_decaying_no_bias_norm_or_embedder(
    tiny_moss_dit: Path,
) -> None:
    transformer = MossTransformer.from_pretrained(str(tiny_moss_dit))

    optimizer = FullOptimizer(transformer, FullSettings(), schedule=lambda done: 1.0)

    muon = {name: group["weight_decay"] for group in optimizer.muon.param_groups for name in group["param_names"]}
    adamw = {name: group["weight_decay"] for group in optimizer.adamw.param_groups for name in group["param_names"]}
    assert muon == dict.fromkeys(JOINT_MATRICES, 0.01)
    # every other weight, decayed only when a matrix of neither an embedder (the latents', prompt's, timestep's) nor a norm
    rest = [name for name, _ in transformer.named_parameters() if name not in JOINT_MATRICES]
    assert adamw == {name: 0.01 if name in DECAYED else 0.0 for name in rest}
    recipe = ("lr", "momentum", "nesterov", "ns_steps", "adjust_lr_fn")
    assert {key: optimizer.muon.param_groups[0][key] for key in recipe} == {
        "lr": 5e-5,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 5,
        "adjust_lr_fn": "match_rms_adamw",
    }
    assert {key: optimizer.adamw.param_groups[0][key] for key in ("lr", "betas", "eps")} == {
        "lr": 5e-5,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }


def test_full_optimizer_decays_the_last_blocks_q_and_k_norm_weights_towards_one(tiny_moss_dit: Path) -> None:
    transformer = MossTransformer.from_pretrained(str(tiny_moss_dit))
    norms = {name: weight for name, weight in transformer.named_parameters() if "norm" in name}
    with torch.no_grad():
        for weight in norms.values():
            weight.fill_(3.0)
    settings = FullSettings(learning_rate=2**-10, weight_decay=2**-4)  # a step of 2^-14 (w - 1), exact in float32
    optimizer = FullOptimizer(transformer, settings, schedule=lambda done: 1.0)

    optimizer.step()  # no gradient: nothing else moves

    moved = {"layers.2.q_norm.weight", "layers.2.k_norm.weight"}
    assert {name: weight.unique().tolist() for name, weight in norms.items()} == {
        name: [3 - 2**-13] if name in moved else [3.0] for name in norms
    }


def test_full_fine_tune_trains_every_weight_of_the_dit_in_float32(
    bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    start = _weights(bfloat16_dit)

    train(bfloat16_dit, tiny_pair_cache, tmp_path, FULL_RUN.model_copy(update={"steps": 1}))

    trained = _weights(_checkpoint(tmp_path, 1) / STATE)
    assert sorted(trained) == sorted(start)
    assert {weight.dtype for weight in trained.values()} == {torch.float32}
    assert [name for name in start if torch.equal(trained[name], start[name].float())] == []


def test_full_checkpoint_is_a_bfloat16_dit_folder_diffu_moss_init_draws_with_its_reference(
    tiny_moss_dit: Path, full_run: Path, tmp_path: Path
) -> None:
    checkpoint = _checkpoint(full_run, 3)
    write_moss_repo(tmp_path / "model", base=tiny_moss_dit.parent, transformer=checkpoint / DIT)

    pipe = MossModularPipeline.from_pretrained(str(tmp_path / "model"))
    pipe.load_components(names=["transformer"])

    trained = _weights(checkpoint / STATE)  # float32: the checkpoint holds them rounded once to bfloat16
    drawn = {name: weight.to(torch.bfloat16) for name, weight in pipe.transformer.state_dict().items()}
    assert (ModelIndex.read(tmp_path / "model").model_extra or {})["takes_reference"] is True
    assert {weight.dtype for weight in _weights(checkpoint / DIT).values()} == {torch.bfloat16}
    assert sorted(drawn) == sorted(trained)
    assert [name for name in trained if not torch.equal(drawn[name], trained[name].to(torch.bfloat16))] == []


def test_full_fine_tune_averages_the_weights_after_each_step_by_the_ema_formula(
    bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    start = {name: weight.float() for name, weight in _weights(bfloat16_dit).items()}

    train(bfloat16_dit, tiny_pair_cache, tmp_path, FULL_RUN.model_copy(update={"steps": 2}))

    weights = [_weights(_checkpoint(tmp_path, step) / STATE) for step in (1, 2)]
    averages = [load_file(str(_checkpoint(tmp_path, step) / STATE / AVERAGE_WEIGHTS)) for step in (1, 2)]
    # average + (1 - decay) * (weight - average), from the weights the run started from
    first = {name: start[name].lerp(weights[0][name], 1 - DECAY) for name in start}
    second = {name: averages[0][name].lerp(weights[1][name], 1 - DECAY) for name in start}
    assert [name for name in start if not torch.equal(averages[0][name], first[name])] == []
    assert [name for name in start if not torch.equal(averages[1][name], second[name])] == []
    drawn = _weights(_checkpoint(tmp_path, 2) / AVERAGE)  # the average as a DiT folder, in bfloat16
    assert [name for name in start if not torch.equal(drawn[name], second[name].to(torch.bfloat16))] == []


def test_full_fine_tune_resumed_from_a_checkpoints_state_goes_on_exactly_as_the_run_went(
    bfloat16_dit: Path, tiny_pair_cache: Path, full_run: Path, tmp_path: Path
) -> None:
    resumed = tmp_path / "run"
    shutil.copytree(full_run, resumed)
    for step in (2, 3):
        shutil.rmtree(_checkpoint(resumed, step))

    train(bfloat16_dit, tiny_pair_cache, resumed, FULL_RUN, resume=_checkpoint(resumed, 1))

    state, ran = _checkpoint(resumed, 3) / STATE, _checkpoint(full_run, 3) / STATE
    weights, expected = _weights(state), _weights(ran)
    averages, expected_averages = load_file(str(state / AVERAGE_WEIGHTS)), load_file(str(ran / AVERAGE_WEIGHTS))
    assert [name for name in expected if not torch.equal(weights[name], expected[name])] == []
    assert [name for name in expected if not torch.equal(averages[name], expected_averages[name])] == []
    logs = [
        [
            {key: value for key, value in json.loads(line).items() if key != "seconds"}
            for line in run.read_text().splitlines()
        ]
        for run in (resumed / LOG, full_run / LOG)
    ]
    assert logs[0] == logs[1]  # the same samples, noise levels, losses and learning rates, steps 1 to 3


def test_full_fine_tune_refuses_to_resume_with_settings_it_was_not_trained_with(
    bfloat16_dit: Path, tiny_pair_cache: Path, full_run: Path, tmp_path: Path
) -> None:
    resumed = tmp_path / "run"
    shutil.copytree(full_run, resumed)
    other = FULL_RUN.model_copy(update={"seed": 1})

    with pytest.raises(ValueError, match="was trained with other settings"):
        train(bfloat16_dit, tiny_pair_cache, resumed, other, resume=_checkpoint(resumed, 1))
