"""The tool environments run our rendered commands (the ``envs`` lane, CPU, the envs built as the README says): a real
``swift sft`` of two steps on a tiny random MiniCPM-OCR (its pinned config shrunk, its own tokenizer and processor)
writes its checkpoint; and speculators, its transformers overridden, reads that tiny MiniCPM-OCR as a verifier in a
dry run of the training command htr-train renders for a draft of it. A missing environment fails the lane rather than
skipping it."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from diffu_htr_train.jobs import RunSpec, ToolEnv
from diffu_htr_train.render import RunPaths, render

if TYPE_CHECKING:
    from conftest import MakeSet

pytestmark = pytest.mark.envs

ENVS = Path(__file__).parents[1] / "envs"
TINY = Path(__file__).parent / "envs_tiny.py"
CPU = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}


def _python(env: ToolEnv) -> Path:
    python = ENVS / env / ".venv" / "bin" / "python"
    if not python.is_file():
        pytest.fail(f"{env} is not built: uv sync --project {ENVS / env}")
    return python


@pytest.fixture(scope="module")
def tiny_minicpm_ocr(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random MiniCPM-OCR folder, built by the minicpm env's own Python."""
    out = tmp_path_factory.mktemp("tiny_minicpm_ocr")
    subprocess.run(
        [str(_python(ToolEnv.MINICPM)), str(TINY), "minicpm-ocr", str(out)], check=True, env=CPU, timeout=900
    )
    return out


def test_swift_sft_trains_a_tiny_minicpm_ocr_two_steps_on_cpu(
    tmp_path: Path, make_set: MakeSet, tiny_minicpm_ocr: Path
) -> None:
    lines = [(f"p_0_{k}", f"Anno {1715 + k}", "GT-A", "V1", "train") for k in range(4)]
    train = make_set("train", lines, size=(224, 32))
    spec = RunSpec.model_validate(
        {
            "name": "tiny",
            "job": {
                "kind": "minicpm-ocr",
                "model": {"repo": str(tiny_minicpm_ocr)},
                "tuner": "full",
                "freeze_vit": False,
                "freeze_aligner": True,
                "freeze_llm": False,
                "torch_dtype": "float32",
                "learning_rate": 1e-4,
                "max_steps": 2,
                "global_batch": 2,
                "max_length": 1024,
                "attn_impl": "sdpa",
                "dataset_num_proc": 1,
                "dataloader_num_workers": 0,
                "save_strategy": "steps",
                "save_steps": 2,
                "logging_steps": 1,
                "report_to": "none",
                "env": {"DOWNSAMPLE_MODE": "4x"},
            },
            "data": {"train": [{"path": str(train)}]},
            "resources": {"need_gb": 8, "hours": 1},
        }
    )
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    [command] = render(spec, paths)

    result = subprocess.run(
        command.argv, env={**CPU, **command.environ}, capture_output=True, text=True, timeout=1800, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert (paths.output / "checkpoint-2").is_dir()


def test_speculators_reads_the_tiny_minicpm_ocr_as_its_verifier(tiny_minicpm_ocr: Path, tmp_path: Path) -> None:
    _python(ToolEnv.SPECULATORS)
    spec = RunSpec.model_validate(
        {
            "name": "tiny-draft",
            "job": {
                "kind": "dspark",
                "target": {"repo": str(tiny_minicpm_ocr)},
                "target_layers": 4,
                "target_layer_ids": [1, 2],
                "num_layers": 1,
                "block_size": 4,
                "responses": str(tmp_path / "responses.jsonl"),
                "media_root": str(tmp_path),
            },
            "resources": {"need_gb": 8, "hours": 1},
        }
    )
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    _, _, train = render(spec, paths)
    (paths.root / "data").mkdir(parents=True)

    result = subprocess.run(
        [*train.argv, "--dry-run"],
        env={**CPU, **train.environ},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
