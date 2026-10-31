"""MOSS-Image's full fine-tune on CPU (conftest's tiny MOSS and tiny style pairs): MOSS's optimizer recipe (Muon on the
joint blocks' matrices, AdamW on the rest, no weight decay on biases, norms and embedders, the last block's q/k norm
weights decayed towards one), Muon's momentum kept in bfloat16 when asked, a step that trains every weight in float32,
a checkpoint that diffu-moss-init draws with, its files byte for byte safetensors' own, the memory of its copies of the
weights given back to the system once each file is written and CUDA's cache emptied after, the average of the weights
by its formula, and a run resumed from a checkpoint's state going on exactly as it would have, and refused one it was
not trained with."""

from __future__ import annotations

import ctypes
import json
import mmap
import shutil
import weakref
from pathlib import Path
from typing import Any, Self

import pytest
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file, save, save_file
from torch import Tensor, nn

from diffu.repo import ModelIndex
from diffu_moss.init import write_moss_repo
from diffu_moss.modular_pipelines.moss import MossModularPipeline
from diffu_moss.train.full import AVERAGE, AVERAGE_WEIGHTS, DIT, STATE, FullOptimizer, FullSettings, MomentumDtype
from diffu_moss.train.loop import LOG, TrainSettings, train
from diffu_moss.train.muon import Muon, orthogonalised, rms_matched_rate
from diffu_moss.train.records import PROGRESS_RECORD
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
FULL = FullSettings(ema=DECAY, save_state=True)
# 3 steps of 2 samples, the learning rate rising over 4 steps (so a resumed run must restore its schedule), an average
# of the weights, a checkpoint and its state after every step
FULL_RUN = TrainSettings(steps=3, save_every=1, warmup=4, full=FULL, device="cpu")


def _weights(folder: Path) -> dict[str, Tensor]:
    """The weights of the tiny DiT folder ``folder`` (one safetensors file), in the dtype they were saved in."""
    return load_file(str(folder / SAFETENSORS_WEIGHTS_NAME))


def _checkpoint(run: Path, step: int) -> Path:
    return run / "checkpoints" / f"step_{step:06d}"


def _weights_files(checkpoint: Path) -> list[Path]:
    """The safetensors files of a checkpoint with an average and a state: the DiT's, the average's, the state's float32
    weights and average."""
    return [
        checkpoint / DIT / SAFETENSORS_WEIGHTS_NAME,
        checkpoint / AVERAGE / SAFETENSORS_WEIGHTS_NAME,
        checkpoint / STATE / SAFETENSORS_WEIGHTS_NAME,
        checkpoint / STATE / AVERAGE_WEIGHTS,
    ]


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

    muon = {
        name: (group["weight_decay"], group["splits"])
        for group in optimizer.muon.param_groups
        for name in group["param_names"]
    }
    adamw = {name: group["weight_decay"] for group in optimizer.adamw.param_groups for name in group["param_names"]}
    # the fused gated feed-forward orthogonalised in halves, every other joint matrix whole
    assert muon == {name: (0.01, 2 if "ffn_up_gate" in name else 1) for name in JOINT_MATRICES}
    # every other weight, decayed only when a matrix of neither an embedder (the latents', prompt's, timestep's) nor a norm
    rest = [name for name, _ in transformer.named_parameters() if name not in JOINT_MATRICES]
    assert adamw == {name: 0.01 if name in DECAYED else 0.0 for name in rest}
    recipe = ("lr", "momentum", "nesterov", "ns_steps")
    assert {key: optimizer.muon.param_groups[0][key] for key in recipe} == {
        "lr": 5e-5,
        "momentum": 0.95,
        "nesterov": True,
        "ns_steps": 5,
    }
    assert optimizer.muon.momentum_dtype == torch.float32  # as torch's Muon keeps it
    assert {key: optimizer.adamw.param_groups[0][key] for key in ("lr", "betas", "eps")} == {
        "lr": 5e-5,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }


def _two_muon_steps(matrix: nn.Parameter, optimizer: Muon, parts: list[nn.Parameter]) -> None:
    """Two steps of ``optimizer`` on ``matrix`` and of torch's Muon (MOSS's recipe) on ``parts``, its row blocks as
    matrices of their own, each step from one random gradient of ``matrix``, its row blocks given to ``parts``."""
    torchs = torch.optim.Muon(
        parts, lr=0.1, weight_decay=0.01, momentum=0.95, nesterov=True, adjust_lr_fn="match_rms_adamw"
    )
    generator = torch.Generator().manual_seed(1)
    for _ in range(2):  # the second step through the momentum
        matrix.grad = torch.randn(matrix.shape, generator=generator)
        for part, grad in zip(parts, matrix.grad.chunk(len(parts)), strict=True):
            part.grad = grad.clone()
        optimizer.step()
        torchs.step()


def test_muon_steps_each_half_of_a_fused_matrix_as_torchs_muon_steps_that_half_alone() -> None:
    fused = nn.Parameter(torch.randn(8, 6, generator=torch.Generator().manual_seed(0)))
    halves = [nn.Parameter(half.detach().clone()) for half in fused.chunk(2)]

    _two_muon_steps(fused, Muon([fused], lr=0.1, weight_decay=0.01, momentum=0.95, splits=2), halves)

    assert torch.equal(fused.detach(), torch.cat([half.detach() for half in halves]))


def test_muon_steps_a_whole_matrix_as_torchs_muon_does() -> None:
    whole = nn.Parameter(torch.randn(6, 4, generator=torch.Generator().manual_seed(0)))  # tall: stepped transposed
    alone = nn.Parameter(whole.detach().clone())

    _two_muon_steps(whole, Muon([whole], lr=0.1, weight_decay=0.01, momentum=0.95), [alone])

    assert torch.equal(whole.detach(), alone.detach())


def test_muon_with_a_bfloat16_momentum_stores_it_rounded_and_steps_from_it_unrounded_in_float32() -> None:
    generator = torch.Generator().manual_seed(0)
    weight = nn.Parameter(torch.randn(4, 6, generator=generator))
    grads = [torch.randn(4, 6, generator=generator) for _ in range(2)]  # the second step from the stored momentum
    expected, momentum = weight.detach().clone(), torch.zeros(4, 6, dtype=torch.bfloat16)
    optimizer = Muon([weight], lr=0.1, weight_decay=0.01, momentum=0.95, momentum_dtype=torch.bfloat16)

    for grad in grads:
        weight.grad = grad.clone()
        optimizer.step()

    # by hand: the momentum 0.95 m + 0.05 g (a lerp) in float32, kept in bfloat16, and the update from it unrounded
    for grad in grads:
        unrounded = momentum.float().lerp(grad, 1 - 0.95)
        momentum = unrounded.bfloat16()
        orthogonal = orthogonalised(grad.lerp(unrounded, 0.95))  # the Nesterov update, in bfloat16 for Newton-Schulz
        expected = expected.mul(1 - 0.1 * 0.01).add(orthogonal, alpha=-rms_matched_rate(0.1, 4, 6))
    stored = optimizer.state_dict()["state"][0]["momentum_buffer"]
    assert stored.dtype == torch.bfloat16
    assert torch.equal(stored, momentum)
    assert torch.equal(weight.detach(), expected)


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


def test_full_checkpoint_writes_each_weights_file_byte_for_byte_as_safetensors_writes_its_weights(
    bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    train(bfloat16_dit, tiny_pair_cache, tmp_path, FULL_RUN.model_copy(update={"steps": 1}))

    checkpoint = _checkpoint(tmp_path, 1)
    trained, averaged = _weights(checkpoint / STATE), load_file(str(checkpoint / STATE / AVERAGE_WEIGHTS))
    expected = [
        save({name: weight.to(torch.bfloat16) for name, weight in trained.items()}),  # each rounded once
        save({name: weight.to(torch.bfloat16) for name, weight in averaged.items()}),
        save(trained),  # the state's, in float32
        save(averaged),
    ]
    files = _weights_files(checkpoint)
    differ = [file for file, data in zip(files, expected, strict=True) if file.read_bytes() != data]
    assert [str(file.relative_to(checkpoint)) for file in differ] == []


def test_full_checkpoint_holds_each_files_weights_in_a_memory_map_of_their_own_unmapped_once_it_is_written(
    bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[weakref.ref[mmap.mmap]] = []  # every memory map made: dead once unmapped
    alive: list[int] = []  # for each map, how many earlier ones were still alive when it was made

    class RecordedMap(mmap.mmap):
        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            alive.append(sum(ref() is not None for ref in made))
            new = super().__new__(cls, *args, **kwargs)
            made.append(weakref.ref(new))
            return new

    held: dict[str, bool] = {}  # each file written: whether a live map held all its tensors

    def recording_save_file(tensors: dict[str, Tensor], filename: str, metadata: dict[str, str] | None = None) -> None:
        maps = [live for live in (ref() for ref in made) if live is not None]
        spans = [(ctypes.addressof(ctypes.c_char.from_buffer(live)), len(live)) for live in maps]
        ends = [(tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in tensors.values()]
        held[filename] = all(
            any(start <= first and last <= start + size for start, size in spans) for first, last in ends
        )
        save_file(tensors, filename, metadata)

    monkeypatch.setattr(mmap, "mmap", RecordedMap)
    monkeypatch.setattr("diffu_moss.train.full.save_file", recording_save_file)

    train(bfloat16_dit, tiny_pair_cache, tmp_path, FULL_RUN.model_copy(update={"steps": 1}))

    files = [str(file) for file in _weights_files(_checkpoint(tmp_path, 1))]
    assert held == dict.fromkeys(files, True)  # each file's tensors in a map
    assert alive == [0] * len(files)  # each map made once every earlier one was unmapped: one file's copies at a time
    assert [ref() for ref in made] == [None] * len(files)  # the last one unmapped too, once its file was written


def test_full_checkpoint_empties_cudas_cache_once_its_files_are_written(
    bfloat16_dit: Path, tiny_pair_cache: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    last = _checkpoint(tmp_path, 1) / STATE / PROGRESS_RECORD
    emptied: list[bool] = []  # each call's: whether the checkpoint's last file was written by then
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(last.exists()))

    train(bfloat16_dit, tiny_pair_cache, tmp_path, FULL_RUN.model_copy(update={"steps": 1}))

    assert emptied.count(True) == 1  # once after it (diffusers empties a visible GPU's cache on loading the DiT too)


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


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"seed": 1}, id="seed"),
        pytest.param(
            {"full": FULL.model_copy(update={"momentum_dtype": MomentumDtype.BFLOAT16})}, id="muon-momentum-dtype"
        ),
    ],
)
def test_full_fine_tune_refuses_to_resume_with_settings_it_was_not_trained_with(
    bfloat16_dit: Path, tiny_pair_cache: Path, full_run: Path, tmp_path: Path, change: dict[str, object]
) -> None:
    resumed = tmp_path / "run"
    shutil.copytree(full_run, resumed)
    other = FULL_RUN.model_copy(update=change)

    with pytest.raises(ValueError, match="was trained with other settings"):
        train(bfloat16_dit, tiny_pair_cache, resumed, other, resume=_checkpoint(resumed, 1))
