"""The tool environments run our rendered commands (the ``envs`` lane, CPU, the envs built as the README says): a real
``swift sft`` of two steps on a tiny random MiniCPM-OCR and on a tiny random LightOnOCR-3 (each its pinned config
shrunk, its own tokenizer and processor) writes its checkpoint; and speculators, its transformers overridden, reads
that tiny MiniCPM-OCR as a verifier in a dry run of the training command htr-train renders for a draft of it. A missing
environment fails the lane rather than skipping it."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from PIL import Image

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


@pytest.fixture(scope="module")
def tiny_lightonocr_3(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny random LightOnOCR-3-4B folder (its own tokenizer, processor and chat template), built by the minicpm
    env's own Python."""
    out = tmp_path_factory.mktemp("tiny_lightonocr_3")
    subprocess.run(
        [str(_python(ToolEnv.MINICPM)), str(TINY), "lightonocr-3", str(out)], check=True, env=CPU, timeout=900
    )
    return out


def _grounded_pages(folder: Path, count: int) -> Path:
    """``count`` pages as ``diffu-reader-set --model lightonocr-3`` writes them: each a 256 x 320 px page (within
    LightOn's 2,048 px) asked ``<image>grounding`` and answered with its blocks (a heading, running text, a table), but
    the last, asked by its image alone (the transcription mode) and answered with the blocks' Markdown."""
    folder.mkdir(parents=True)
    rows = []
    for k in range(count):
        page = folder / f"p{k}.jpg"
        Image.new("RGB", (256, 320), (40 * k % 256,) * 3).save(page)
        blocks = [
            ("title", "100,50,900,100", "## Förteckning öfver gods"),
            ("text", "100,120,900,280", f"Anno {1715 + k} den {k + 1} maj\nkom för rätten"),
            ("table", "100,300,900,400", "<table><tr><td>Namn</td><td>Ort</td></tr></table>"),
        ]
        grounding = k < count - 1
        answer = "\n\n".join(f"![{label}]({box})\n{text}" if grounding else text for label, box, text in blocks)
        prompt = "<image>grounding" if grounding else "<image>"
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        rows.append({"messages": messages, "images": [str(page)]})
    path = folder / "pages.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def _tiny_spec(model: Path, train: Path) -> RunSpec:
    """A LightOnOCR-3 run of two steps on CPU (float32) of ``model`` on ``train``, checkpointed at its end."""
    return RunSpec.model_validate(
        {
            "name": "tiny-lightonocr-3",
            "job": {
                "kind": "lightonocr-3",
                "model": {"repo": str(model)},
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
            },
            "data": {"train": [{"path": str(train)}]},
            "resources": {"need_gb": 8, "hours": 1},
        }
    )


def test_the_rendered_lightonocr_3_command_has_ms_swift_encode_a_page_as_lightonocr_3s_own_chat_template(
    tmp_path: Path, tiny_lightonocr_3: Path
) -> None:
    train = _grounded_pages(tmp_path / "sets", 3)
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    [command] = render(_tiny_spec(tiny_lightonocr_3, train), paths)
    template_flags = ("--model_type", "--template", "--add_non_thinking_prefix", "--loss_scale")
    rendered = [word for flag in template_flags for word in (flag, command.argv[command.argv.index(flag) + 1])]

    result = subprocess.run(
        [
            str(_python(ToolEnv.MINICPM)),
            str(Path(__file__).parent / "envs_template.py"),
            *(str(tiny_lightonocr_3), str(train)),
            *rendered,
        ],
        env={**CPU, **command.environ},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    same = {"same_ids": True, "same_grid": True, "loss_from_generation_prompt_end": True}
    assert [json.loads(line) for line in result.stdout.splitlines()] == [same] * 3


def test_swift_sft_trains_a_tiny_lightonocr_3_two_steps_on_cpu_with_its_loss_on_the_answers(
    tmp_path: Path, tiny_lightonocr_3: Path
) -> None:
    paths = RunPaths(root=tmp_path / "run", envs=ENVS.resolve(), trackio_project="")
    [command] = render(_tiny_spec(tiny_lightonocr_3, _grounded_pages(tmp_path / "sets", 4)), paths)

    result = subprocess.run(
        command.argv, env={**CPU, **command.environ}, capture_output=True, text=True, timeout=1800, check=False
    )

    assert result.returncode == 0, result.stderr[-4000:]
    state = json.loads((paths.output / "checkpoint-2" / "trainer_state.json").read_text(encoding="utf-8"))
    losses = [entry["loss"] for entry in state["log_history"] if "loss" in entry]
    assert len(losses) == 2
    assert all(math.isfinite(loss) and loss > 0 for loss in losses)


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
