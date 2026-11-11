"""Nemotron 3.5 Parse trained by our trainer in its environment, on CPU, as htr-train renders the run (the envs lane: the
model at its pinned revision, its code, processor and tokenizer; small drawn pages): two steps in two processes under
torchrun (DDP over gloo) leave a checkpoint that reads as the model's repo does, its frozen parts as shipped and its
decoder trained, and the run's model in bfloat16; a stopped run is resumed by running its steps again; and a batch of
pages of other sizes and answers of other lengths gives each page the loss it has alone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from safetensors import safe_open

from diffu_htr_train.jobs import RunSpec, ToolEnv, Where
from diffu_htr_train.launch import steps_script
from diffu_htr_train.nemotron_parse import REPO, REVISION, USER_TURN
from diffu_htr_train.render import Command, RunPaths, render

pytestmark = pytest.mark.envs

ENVS = Path(__file__).parents[1] / "envs"
PYTHON = ENVS / ToolEnv.NEMOTRON_PARSE / ".venv" / "bin" / "python"
BATCH = Path(__file__).parent / "nemotron_parse_batch.py"
CPU = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
# pages of three sizes (W x H), each with two lines of text and its answer in the model's block form
PAGES = [((192, 256), "Anno 1715", "den 4 maj"), ((320, 224), "kom för rätten", "Anders"), ((256, 320), "Summa", "2")]
MTP = "decoder.extra_heads.0.weight"
VISION = "encoder.model_encoder.radio_model.model.blocks.0.attn.qkv.weight"
DECODER = "decoder.layers.0.fc1.weight"


@pytest.fixture(scope="module", autouse=True)
def _built() -> None:
    if not PYTHON.is_file():
        pytest.fail(f"envs/nemotron_parse is not built: uv sync --project {PYTHON.parents[2]}")


def _page_set(folder: Path, name: str, volume: str) -> Path:
    """A set of :data:`PAGES` as diffu-reader-set writes Nemotron-Parse's (each page asked its task prompt and answered
    with one text block), real training pages of ``volume``, its meta beside it."""
    folder.mkdir(parents=True, exist_ok=True)
    rows, metas = [], []
    for k, ((width, height), first, second) in enumerate(PAGES):
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((16, 24), first, fill="black")
        draw.text((16, 48), second, fill="black")
        path = folder / f"{name}_{k}.jpg"
        image.save(path)
        answer = f"<x_0.083><y_0.0938>{first}\n{second}<x_0.75><y_0.25><class_Text>"
        rows.append(
            {
                "messages": [{"role": "user", "content": USER_TURN}, {"role": "assistant", "content": answer}],
                "images": [str(path.resolve())],
            }
        )
        metas.append({"id": f"{name}_{k}", "source": "real", "collection": "GT-A", "volume": volume, "split": "train"})
    path = folder / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (folder / f"{name}.meta.jsonl").write_text("".join(json.dumps(meta) + "\n" for meta in metas), encoding="utf-8")
    return path


def _spec(sets: Path, gpus: int, **job: Any) -> RunSpec:
    """A small run of our trainer: two steps of one page a process, the vision encoder frozen, the rest trained."""
    return RunSpec.model_validate(
        {
            "name": "tiny-np",
            "job": {
                "kind": "nemotron-parse",
                "model": {"repo": REPO, "revision": REVISION},
                "freeze_vision_encoder": True,
                "freeze_adapter": False,
                "activation_checkpointing": True,
                "bf16": False,
                "learning_rate": 1e-4,
                "max_steps": 2,
                "global_batch": gpus,
                "dataloader_num_workers": 0,
                "eval_strategy": "steps",
                "eval_steps": 2,
                "save_strategy": "steps",
                "save_steps": 1,
                "logging_steps": 1,
                "ddp_find_unused_parameters": False,
                "report_to": "none",
                **job,
            },
            "data": {
                "train": [{"path": str(_page_set(sets, "train", "V1"))}],
                "dev": str(_page_set(sets, "dev", "V2")),
            },
            "resources": {"gpus_per_node": gpus, "need_gb": 16, "hours": 1},
        }
    )


def _on_cpu(command: Command) -> Command:
    """``command`` on the CPU: transformers' own switch, which runs DDP over gloo under torchrun."""
    return command.model_copy(update={"argv": [*command.argv, "--use_cpu", "true"]})


def _tensor(path: Path, name: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt") as weights:
        return weights.get_tensor(name)


def test_two_processes_train_the_model_and_save_it_as_its_repo_frozen_parts_as_shipped(tmp_path: Path) -> None:
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    [command] = render(_spec(tmp_path / "sets", gpus=2), paths)

    result = subprocess.run(_on_cpu(command).argv, env=CPU, capture_output=True, text=True, timeout=1800, check=False)

    assert result.returncode == 0, result.stderr[-4000:]
    checkpoint = paths.output / "checkpoint-2"
    assert json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))["global_step"] == 2
    # the checkpoint reads as the repo does: its architecture's name, code, processor and tokenizer beside its weights
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    assert config["architectures"] == ["RadioMBartPrefixForCausalLM"]
    assert {"hf_nemotron_parse_modeling.py", "preprocessor_config.json", "tokenizer.json"} <= {
        path.name for path in checkpoint.iterdir()
    }
    # its frozen parts (the vision encoder, the MTP head) as shipped, its decoder trained
    shipped = (
        Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["model.safetensors"])) / "model.safetensors"
    )
    trained = checkpoint / "model.safetensors"
    assert torch.equal(_tensor(trained, VISION), _tensor(shipped, VISION).float())
    assert torch.equal(_tensor(trained, MTP), _tensor(shipped, MTP).float())
    assert not torch.equal(_tensor(trained, DECODER), _tensor(shipped, DECODER).float())
    # the run's model in bfloat16, as the repo ships it, named as the repo names it
    final = json.loads((paths.output / "config.json").read_text(encoding="utf-8"))
    assert (final["architectures"], _tensor(paths.output / "model.safetensors", DECODER).dtype) == (
        ["RadioMBartPrefixForCausalLM"],
        torch.bfloat16,
    )


def test_a_stopped_run_resumes_from_its_newest_checkpoint_when_its_steps_run_again(tmp_path: Path) -> None:
    spec = _spec(tmp_path / "sets", gpus=1)
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    commands = [_on_cpu(command) for command in render(spec, paths)]
    steps = tmp_path / "run" / "steps.sh"
    steps.parent.mkdir(parents=True)
    steps.write_text(
        steps_script(spec, where=Where.LOCAL, commands=commands, paths=paths, setup=["export CUDA_VISIBLE_DEVICES="]),
        encoding="utf-8",
    )
    first = subprocess.run(["bash", str(steps)], env=CPU, capture_output=True, text=True, timeout=1800, check=False)
    assert first.returncode == 0, first.stderr[-4000:]
    shutil.rmtree(paths.output / "checkpoint-2")  # as if it had stopped before its second checkpoint

    second = subprocess.run(["bash", str(steps)], env=CPU, capture_output=True, text=True, timeout=1800, check=False)

    assert second.returncode == 0, second.stderr[-4000:]
    assert f"htr-train: resuming from {paths.output.resolve() / 'checkpoint-1'}" in second.stderr
    state = json.loads((paths.output / "checkpoint-2" / "trainer_state.json").read_text(encoding="utf-8"))
    # the second step trained once: the first run's first step, then the resumed run's second
    assert [entry["step"] for entry in state["log_history"] if "loss" in entry] == [1, 2]


def test_a_batch_of_pages_of_other_sizes_gives_each_page_the_loss_it_has_alone(tmp_path: Path) -> None:
    pages = _page_set(tmp_path / "sets", "pages", "V1")

    result = subprocess.run(
        [str(PYTHON), "-I", str(BATCH), str(pages)], env=CPU, capture_output=True, text=True, timeout=1800, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    read = json.loads(result.stdout.splitlines()[-1])
    assert read["sizes"] == [list(size) for size, _, _ in PAGES]
    assert read["batched"] == pytest.approx(read["alone"], rel=1e-4)
