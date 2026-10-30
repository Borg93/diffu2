"""MOSS-Image's full fine-tune over several processes on CPU (gloo; conftest's tiny MOSS and tiny style pairs): FSDP2
launched by torchrun at world size 2 trains, scores, checkpoints and resumes exactly as at world size 1 for the same
global batch, Muon's momentum kept in bfloat16 too, and a step's samples must share out evenly among the ranks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME
from safetensors.torch import load_file
from torch import Tensor

from diffu_moss.train.distributed import Ranks
from diffu_moss.train.full import AVERAGE, AVERAGE_WEIGHTS, DIT, OPTIMIZER_STATE, STATE, FullSettings
from diffu_moss.train.loop import LOG, TrainSettings, train

STEPS = 3


def _torchrun(world: int, run: Path, dit: Path, cache: Path, *options: str) -> None:
    """A full fine-tune of ``dit`` on ``cache`` (scored on it too) launched by torchrun on ``world`` CPU processes: 3
    steps of 2 samples, an average of the weights, a checkpoint with its state after every step."""
    command = [
        *(sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"),
        *("-m", "diffu_moss.train.loop", "--full", "--device", "cpu"),
        *("--transformer", str(dit), "--cache", str(cache), "--val-cache", str(cache), "--out", str(run)),
        *("--steps", str(STEPS), "--samples-per-step", "2", "--save-every", "1", "--val-every", str(STEPS)),
        *("--learning-rate", "1e-3", "--ema", "0.9", "--save-state", *options),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    done = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr[-3000:]


def _log(run: Path) -> list[dict[str, object]]:
    return [
        {key: value for key, value in json.loads(line).items() if key not in ("seconds", "peak_memory_gb")}
        for line in (run / LOG).read_text(encoding="utf-8").splitlines()
    ]


def _checkpoint(run: Path, step: int) -> Path:
    return run / "checkpoints" / f"step_{step:06d}"


def _tensors(checkpoint: Path) -> dict[str, Tensor]:
    """Every tensor a checkpoint holds: its bfloat16 DiT and average, and its state's float32 weights and average."""
    files = {
        "dit": checkpoint / DIT / SAFETENSORS_WEIGHTS_NAME,
        "ema": checkpoint / AVERAGE / SAFETENSORS_WEIGHTS_NAME,
        "state": checkpoint / STATE / SAFETENSORS_WEIGHTS_NAME,
        "state_ema": checkpoint / STATE / AVERAGE_WEIGHTS,
    }
    return {f"{kind}/{name}": tensor for kind, file in files.items() for name, tensor in load_file(str(file)).items()}


def _optimizer_tensors(checkpoint: Path) -> dict[str, Tensor]:
    saved = torch.load(checkpoint / STATE / OPTIMIZER_STATE, weights_only=True)
    return {
        f"{optimizer}/{index}/{key}": value
        for optimizer in ("muon", "adamw")
        for index, values in saved[optimizer]["state"].items()
        for key, value in values.items()
    }


@pytest.fixture(scope="module")
def world_runs(tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """The same full fine-tune at world size 1 and 2: each run's directory."""
    root = tmp_path_factory.mktemp("world_runs")
    for world in (1, 2):
        _torchrun(world, root / f"world_{world}", tiny_moss_dit, tiny_pair_cache)
    return {world: root / f"world_{world}" for world in (1, 2)}


def test_full_fine_tune_at_world_size_2_logs_trains_and_checkpoints_exactly_as_at_world_size_1(
    world_runs: dict[int, Path],
) -> None:
    one, two = world_runs[1], world_runs[2]

    logs = _log(one), _log(two)

    assert logs[0] == logs[1]  # the samples, noise levels, losses, gradient norms and validation losses
    for step in range(1, STEPS + 1):
        tensors = _tensors(_checkpoint(one, step)), _tensors(_checkpoint(two, step))
        assert sorted(tensors[0]) == sorted(tensors[1])
        assert [name for name in tensors[0] if not torch.equal(tensors[0][name], tensors[1][name])] == []
        states = _optimizer_tensors(_checkpoint(one, step)), _optimizer_tensors(_checkpoint(two, step))
        assert sorted(states[0]) == sorted(states[1])
        assert [name for name in states[0] if not torch.equal(states[0][name], states[1][name])] == []


def test_full_fine_tune_resumed_at_world_size_2_from_a_world_size_1_state_goes_on_exactly_as_the_run_went(
    tiny_moss_dit: Path, tiny_pair_cache: Path, world_runs: dict[int, Path], tmp_path: Path
) -> None:
    resumed = tmp_path / "run"
    shutil.copytree(world_runs[1], resumed)
    for step in range(2, STEPS + 1):
        shutil.rmtree(_checkpoint(resumed, step))

    _torchrun(2, resumed, tiny_moss_dit, tiny_pair_cache, "--resume", str(_checkpoint(resumed, 1)))

    assert _log(resumed) == _log(world_runs[1])
    tensors = _tensors(_checkpoint(resumed, STEPS)), _tensors(_checkpoint(world_runs[1], STEPS))
    assert [name for name in tensors[1] if not torch.equal(tensors[0][name], tensors[1][name])] == []


def test_full_fine_tune_with_a_bfloat16_muon_momentum_resumed_at_world_size_2_goes_on_exactly_as_at_world_size_1(
    tiny_moss_dit: Path, tiny_pair_cache: Path, tmp_path: Path
) -> None:
    ran, resumed = tmp_path / "world_1", tmp_path / "resumed"
    bfloat16 = ("--muon-momentum-dtype", "bfloat16")
    _torchrun(1, ran, tiny_moss_dit, tiny_pair_cache, *bfloat16)
    shutil.copytree(ran, resumed)
    for step in range(2, STEPS + 1):
        shutil.rmtree(_checkpoint(resumed, step))

    _torchrun(2, resumed, tiny_moss_dit, tiny_pair_cache, *bfloat16, "--resume", str(_checkpoint(resumed, 1)))

    assert _log(resumed) == _log(ran)
    tensors = _tensors(_checkpoint(resumed, STEPS)), _tensors(_checkpoint(ran, STEPS))
    assert [name for name in tensors[1] if not torch.equal(tensors[0][name], tensors[1][name])] == []
    states = _optimizer_tensors(_checkpoint(resumed, STEPS)), _optimizer_tensors(_checkpoint(ran, STEPS))
    # the momentum of each of the 18 joint-block matrices saved in bfloat16, read back, stepped and saved so again
    momenta = [[state.dtype for name, state in saved.items() if name.startswith("muon/")] for saved in states]
    assert momenta == [[torch.bfloat16] * 18] * 2
    assert sorted(states[0]) == sorted(states[1])
    assert [name for name in states[1] if not torch.equal(states[0][name], states[1][name])] == []


def test_training_refuses_samples_a_step_that_do_not_share_out_evenly_among_the_ranks(
    tiny_pair_cache: Path, tmp_path: Path
) -> None:
    settings = TrainSettings(steps=1, grad_accumulation=3, full=FullSettings(), device="cpu")

    with pytest.raises(ValueError, match="3 samples a step do not share out among 2 ranks"):
        train(tmp_path / "no_dit", tiny_pair_cache, tmp_path / "run", settings, ranks=Ranks(rank=0, world=2))
