"""A planned run is written as its folder (the spec, a manifest of what it ran from, its scripts) and started: on the
Spark its steps run under the memory guard with the guard's floor and the run's memory, the tool called in its own
environment; on Slurm an sbatch file runs one task a node with ms-swift's multi-node variables; a draft's served model
is started, waited for and stopped around its steps, and a target that dies before it answers fails them; a queue runs
Spark runs one after another, waiting out a busy GPU; a local run takes the server's GPUs its settings name, as many
as it asks for, with ms-swift's process a GPU; the smoke command writes each smoke run and queues them; a queued Slurm
run and a served target for a run that serves none are refused."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from pydantic import JsonValue
from typer.testing import CliRunner

from diffu_htr_train.cli import app
from diffu_htr_train.jobs import JobRefused, RunSpec, ToolEnv, Where
from diffu_htr_train.launch import Plan, plan_run, queue_script, write_run
from diffu_htr_train.settings import Settings

if TYPE_CHECKING:
    from conftest import MakeSet

MINICPM: dict[str, JsonValue] = {
    "repo": "openbmb/MiniCPM-OCR-1.0",
    "revision": "2697d4e93dca82784abf858ae57206a49629cb6b",
}


def _job() -> dict[str, JsonValue]:
    return {
        "kind": "minicpm-ocr",
        "model": dict(MINICPM),
        "tuner": "full",
        "freeze_vit": False,
        "freeze_aligner": True,
        "learning_rate": 5e-6,
        "max_steps": 2,
        "global_batch": 8,
        "max_length": 4096,
        "env": {"DOWNSAMPLE_MODE": "4x"},
    }


def _spec(make_set: MakeSet, **resources: JsonValue) -> RunSpec:
    train = make_set("train", [("a_0_0", "Anno", "GT-A", "V1", "train")])
    return RunSpec.model_validate(
        {
            "name": "lines",
            "description": "two steps",
            "job": _job(),
            "data": {"train": [{"path": str(train)}]},
            "resources": {"need_gb": 40, "hours": 1.5, **resources},
        }
    )


def _executable(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _environ(settings: Settings) -> dict[str, str]:
    """``settings``' machine paths as the CLI reads them, from the environment."""
    return {
        "HTR_TRAIN_RUNS": str(settings.runs),
        "HTR_TRAIN_ENVS": str(settings.envs),
        "HTR_TRAIN_GUARD": str(settings.guard),
        "HTR_TRAIN_FLOOR_FILE": str(settings.floor_file),
        "HTR_TRAIN_QUEUE": str(settings.queue),
    }


def _written(spec: RunSpec, path: Path) -> Path:
    """``spec`` written as the YAML file a user writes, at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec.model_dump(mode="json")), encoding="utf-8")
    return path


def test_launch_runs_the_rendered_command_in_its_environment_under_the_memory_guard(
    tmp_path: Path, make_set: MakeSet, settings: Settings
) -> None:
    spec = _spec(make_set)
    spec_file = _written(spec, tmp_path / "lines.yaml")
    calls = tmp_path / "calls"
    # stand-ins for the guard (it records its options and runs what follows --) and for ms-swift (it records its argv
    # and the environment it was given)
    _executable(
        settings.guard,
        f'#!/bin/bash\necho "guard $*" >> {calls}\nwhile [ "$1" != "--" ]; do shift; done\nshift\nexec "$@"\n',
    )
    _executable(
        settings.envs / "minicpm" / ".venv" / "bin" / "swift",
        '#!/bin/bash\necho "$0 $* DOWNSAMPLE_MODE=$DOWNSAMPLE_MODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"'
        f" >> {calls}\n",
    )

    result = CliRunner().invoke(app, ["launch", str(spec_file), "--where", "spark"], env=_environ(settings))

    root = (settings.runs / "lines").resolve()
    plan = plan_run(spec, Where.SPARK, settings)
    assert result.exit_code == 0, result.output
    assert calls.read_text(encoding="utf-8").splitlines() == [
        f"guard --need-gb 40 --floor-gb 15 --name htr-lines --log {root / 'guard.log'} -- bash {root / 'steps.sh'}",
        f"{' '.join(plan.commands[0].argv)} DOWNSAMPLE_MODE=4x CUDA_VISIBLE_DEVICES=0",
    ]
    assert sorted(path.name for path in root.iterdir()) == ["command.sh", "job.yaml", "manifest.json", "steps.sh"]
    assert RunSpec.model_validate(yaml.safe_load((root / "job.yaml").read_text(encoding="utf-8"))) == spec


def test_write_run_records_what_the_run_ran_from_in_its_manifest(
    tmp_path: Path, make_set: MakeSet, settings: Settings
) -> None:
    spec = _spec(make_set)
    plan = plan_run(spec, Where.SPARK, settings)
    lock = settings.envs / "minicpm" / "uv.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("version = 1\n", encoding="utf-8")
    train = tmp_path / "sets" / "train.jsonl"
    (tmp_path / "sets" / "train.options.json").write_text('{"task": "lines"}\n', encoding="utf-8")

    root = write_run(plan)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert {key: manifest[key] for key in ("name", "description", "kind", "where", "env_locks", "models")} == {
        "name": "lines",
        "description": "two steps",
        "kind": "minicpm-ocr",
        "where": "spark",
        "env_locks": {"minicpm": "dbab12665d98aef021ba64953c61b0ed8a908cfb56a1c01e2fcb4b052b71a2a1"},
        "models": ["openbmb/MiniCPM-OCR-1.0@2697d4e93dca82784abf858ae57206a49629cb6b"],
    }
    assert [(row["path"], row["lines"], row["options"]) for row in manifest["data"]] == [
        (str(train.resolve()), 1, {"task": "lines"}),
        (str(train.with_name("train.meta.jsonl").resolve()), 1, None),
    ]
    assert manifest["commands"] == [command.model_dump(mode="json") for command in plan.commands]
    assert (len(manifest["git_commit"]), type(manifest["git_dirty"])) == (40, bool)


def test_write_run_refuses_a_run_name_already_run_and_leaves_that_run_as_it_was(
    make_set: MakeSet, settings: Settings
) -> None:
    plan = plan_run(_spec(make_set), Where.SPARK, settings)
    root = write_run(plan)
    before = {path.name: path.read_bytes() for path in root.iterdir()}

    with pytest.raises(JobRefused, match="holds a run already"):
        write_run(plan)

    assert {path.name: path.read_bytes() for path in root.iterdir()} == before


def test_plan_run_writes_an_sbatch_of_one_task_a_node_with_ms_swifts_multi_node_variables(
    make_set: MakeSet, settings: Settings
) -> None:
    spec = _spec(make_set, nodes=2, gpus_per_node=4, cpus_per_task=32)
    site = settings.model_copy(update={"slurm_partition": "boost_usr_prod", "slurm_account": "proj"})

    plan = plan_run(spec, Where.SLURM, site)

    root = (settings.runs / "lines").resolve()
    assert plan.scripts["job.sbatch"] == (
        "#!/bin/bash\n"
        "#SBATCH --job-name=htr-lines\n"
        "#SBATCH --nodes=2\n"
        "#SBATCH --ntasks-per-node=1\n"
        "#SBATCH --gpus-per-node=4\n"
        "#SBATCH --cpus-per-task=32\n"
        "#SBATCH --mem=0\n"
        "#SBATCH --time=01:30:00\n"
        f"#SBATCH --output={root / 'slurm-%j.out'}\n"
        "set -euo pipefail\n"
        f"srun bash {root / 'steps.sh'}\n"
    )
    steps = plan.scripts["steps.sh"].splitlines()
    assert steps[3:7] == [
        "export NNODES=$SLURM_NNODES NODE_RANK=$SLURM_NODEID",
        'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"',
        "export MASTER_PORT=29500",
        "export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8",
    ]
    assert steps[-1] == f'env DOWNSAMPLE_MODE=4x NPROC_PER_NODE=4 {shlex.join(plan.commands[0].argv)} "${{RESUME[@]}}"'
    assert plan.start == ["sbatch", "--partition", "boost_usr_prod", "--account", "proj", str(root / "job.sbatch")]


def _draft(tmp_path: Path, settings: Settings) -> Plan:
    """A draft's run planned on the Spark, on responses its target answered, its runs folder given relative (as its
    default is) from ``tmp_path``."""
    target = tmp_path / "target"
    target.mkdir()
    responses = tmp_path / "responses.jsonl"
    responses.write_text("{}\n", encoding="utf-8")
    record = {"target": str(target.resolve()), "samples": 1, "failed": 0, "max_failed": 0.01, "max_tokens": 512}
    (tmp_path / "responses.target.json").write_text(json.dumps(record), encoding="utf-8")
    job = {"kind": "dspark", "target": {"repo": str(target)}, "responses": str(responses), "media_root": "."}
    spec = RunSpec.model_validate({"name": "draft", "job": job, "resources": {"need_gb": 60, "hours": 1}})
    return plan_run(spec, Where.SPARK, settings.model_copy(update={"runs": Path("runs")}))


def _run_steps(plan: Plan, tmp_path: Path, *, vllm: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """The plan's steps.sh run with stand-ins on its tools' paths: vllm (the script ``vllm``), curl (on ``PATH``,
    answering once the stand-in vllm has recorded its start, after a first failure) and speculators
    (recording their argv); the run's result and the calls recorded, in order."""
    calls, tried = tmp_path / "calls", tmp_path / "tried"
    recording = f'#!/bin/bash\necho "$(basename "$0") $*" >> {calls}\n'
    _executable(Path(plan.paths.tool(ToolEnv.VLLM, "vllm")), vllm.format(calls=calls))
    _executable(Path(plan.paths.tool(ToolEnv.SPECULATORS, "speculators")), recording)
    _executable(
        tmp_path / "bin" / "curl",
        f'#!/bin/bash\necho "curl $*" >> {calls}\n[ -f {tried} ] && grep -q "^vllm " {calls} && exit 0\n'
        f"touch {tried}\nexit 1\n",
    )
    steps = plan.paths.root / "steps.sh"
    steps.parent.mkdir(parents=True)
    steps.write_text(plan.scripts["steps.sh"], encoding="utf-8")
    environ = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(["bash", str(steps)], env=environ, capture_output=True, text=True, timeout=60, check=False)

    return result, calls.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize(
    ("checkpoints", "save_only_model", "resumed_from"),
    [
        ([], None, None),  # a first start
        (["checkpoint-2", "checkpoint-10", "checkpoint-11/partial"], None, "checkpoint-10"),  # 11: crashed saving
        (["checkpoint-10"], True, None),  # model-only checkpoints: no optimizer state to resume from
    ],
)
def test_steps_run_again_resume_a_training_from_its_newest_whole_checkpoint_without_training_its_batches_again(
    tmp_path: Path,
    make_set: MakeSet,
    settings: Settings,
    checkpoints: list[str],
    save_only_model: bool | None,
    resumed_from: str | None,
) -> None:
    given = _spec(make_set).model_dump(mode="json")
    spec = RunSpec.model_validate({**given, "job": {**given["job"], "save_only_model": save_only_model}})
    plan = plan_run(spec, Where.LOCAL, settings.model_copy(update={"local_gpus": "1"}))
    calls = tmp_path / "calls"
    _executable(Path(plan.paths.tool(ToolEnv.MINICPM, "swift")), f'#!/bin/bash\necho "$*" >> {calls}\n')
    output = plan.paths.output.resolve()
    for checkpoint in checkpoints:
        whole = not checkpoint.endswith("/partial")
        folder = output / checkpoint.removesuffix("/partial")
        folder.mkdir(parents=True)
        if whole:
            (folder / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    steps = plan.paths.root / "steps.sh"
    steps.parent.mkdir(parents=True, exist_ok=True)
    steps.write_text(plan.scripts["steps.sh"], encoding="utf-8")

    result = subprocess.run(["bash", str(steps)], capture_output=True, text=True, timeout=60, check=False)

    resume = ["--resume_from_checkpoint", str(output / resumed_from)] if resumed_from else []
    assert result.returncode == 0, result.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [" ".join([*plan.commands[0].argv[1:], *resume])]
    assert "--ignore_data_skip" not in plan.scripts["steps.sh"]


def test_steps_start_a_drafts_served_target_wait_for_its_health_and_stop_it_after_the_steps(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = _draft(tmp_path, settings)
    _, prepare, train = plan.commands

    result, calls = _run_steps(plan, tmp_path, vllm='#!/bin/bash\necho "vllm $$" >> {calls}\nexec sleep 30\n')

    health = "curl -sf http://localhost:8000/health"
    served = next(k for k, call in enumerate(calls) if call.startswith("vllm "))
    asked = [k for k, call in enumerate(calls) if call.startswith("curl ")]
    assert result.returncode == 0, result.stderr
    assert (
        [call for call in calls if not call.startswith("vllm ")]
        == [  # failed once, answered, checked
            health,
            health,
            health,
            " ".join(["speculators", *prepare.argv[1:]]),
            " ".join(["speculators", *train.argv[1:]]),
        ]
    )
    assert served < asked[1]
    assert (tmp_path / "runs" / "draft" / "serve.log").is_file()
    with pytest.raises(ProcessLookupError):
        os.kill(int(calls[served].split()[1]), 0)


def test_steps_fail_when_a_drafts_served_target_dies_before_it_answers(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    plan = _draft(tmp_path, settings)

    result, calls = _run_steps(plan, tmp_path, vllm="#!/bin/bash\nexit 1\n")

    assert (result.returncode, result.stderr) == (1, "the served model stopped: see its log\n")
    assert {call.split()[0] for call in calls} == {"curl"}


def test_queue_script_runs_one_after_another_and_waits_out_a_busy_gpu(
    tmp_path: Path, make_set: MakeSet, settings: Settings
) -> None:
    plan = plan_run(_spec(make_set), Where.SPARK, settings)
    tries = tmp_path / "tries"
    busy_once = _executable(  # the guard's exit 3 the first time (another guarded job holds the GPU), then 0
        tmp_path / "busy_once.sh",
        f'#!/bin/bash\necho try >> {tries}\n[ "$(wc -l < {tries})" -ge 2 ]\nstatus=$?\n[ $status -eq 0 ] || exit 3\n',
    )
    queue = tmp_path / "queue.sh"
    plan = plan.model_copy(update={"start": ["bash", str(busy_once)]})
    queue.write_text(queue_script([plan]).replace("sleep 300", "sleep 0"), encoding="utf-8")

    status = subprocess.run(["bash", str(queue)], check=False, timeout=60).returncode

    assert (status, tries.read_text(encoding="utf-8").splitlines()) == (0, ["try", "try"])


def test_plan_run_gives_slurm_a_time_past_a_day_in_hours(make_set: MakeSet, settings: Settings) -> None:
    plan = plan_run(_spec(make_set, hours=25.0), Where.SLURM, settings)

    assert [line for line in plan.scripts["job.sbatch"].splitlines() if line.startswith("#SBATCH --time")] == [
        "#SBATCH --time=25:00:00"
    ]


def test_render_refuses_a_run_whose_sets_are_missing_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # short paths: rich wraps its output at 80 columns
    spec = {
        "name": "lines",
        "job": _job(),
        "data": {"train": [{"path": "missing.jsonl"}]},
        "resources": {"need_gb": 40, "hours": 1},
    }
    Path("spec.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = CliRunner().invoke(app, ["render", "spec.yaml"], env={"HTR_TRAIN_RUNS": "runs"})

    assert (result.exit_code, Path("runs").exists()) == (1, False)
    assert "refused: missing.jsonl does not exist" in result.output


def test_smoke_writes_each_smoke_run_and_queues_them_in_the_settings_folders(
    tmp_path: Path, make_set: MakeSet, settings: Settings
) -> None:
    configs = _written(_spec(make_set), tmp_path / "smoke" / "lines.yaml").parent

    result = CliRunner().invoke(app, ["smoke", str(configs)], env=_environ(settings))

    root = (settings.runs / "lines").resolve()
    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in root.iterdir()) == ["command.sh", "job.yaml", "manifest.json", "steps.sh"]
    assert (settings.queue / "htr_smoke.sh").read_text(encoding="utf-8").splitlines()[-1] == (
        f"guarded bash {root / 'command.sh'}"
    )


def test_launch_refuses_to_queue_a_slurm_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Path("spec.yaml").write_text("{}\n", encoding="utf-8")

    result = CliRunner().invoke(app, ["launch", "spec.yaml", "--where", "slurm", "--queue"])

    assert result.exit_code == 1
    assert "refused: --queue writes the Spark's guarded queue: on slurm, sbatch queues the run itself" in " ".join(
        result.output.split()
    )


def test_serve_target_refuses_a_run_that_serves_no_target(
    tmp_path: Path, make_set: MakeSet, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(make_set)
    monkeypatch.chdir(tmp_path)
    _written(spec, Path("spec.yaml"))

    result = CliRunner().invoke(app, ["serve-target", "spec.yaml"])

    assert result.exit_code == 1
    assert "refused: spec.yaml is a minicpm-ocr run: only a dspark run serves a target" in " ".join(
        result.output.split()
    )


def test_plan_run_runs_a_local_run_on_the_gpus_its_settings_name_with_a_process_a_gpu(
    make_set: MakeSet, settings: Settings
) -> None:
    spec = _spec(make_set, gpus_per_node=2)

    plan = plan_run(spec, Where.LOCAL, settings.model_copy(update={"local_gpus": "1, 2"}))

    root = (settings.runs / "lines").resolve()
    steps = plan.scripts["steps.sh"].splitlines()
    assert steps[3:6] == [
        "export CUDA_VISIBLE_DEVICES=1,2",
        "export OMP_NUM_THREADS=8",
        'export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"',
    ]
    assert plan.commands[0].environ["NPROC_PER_NODE"] == "2"
    assert "--gradient_accumulation_steps 4" in shlex.join(plan.commands[0].argv)
    assert plan.scripts["command.sh"] == f"#!/bin/bash\nset -euo pipefail\nexec bash {root / 'steps.sh'}\n"
    assert plan.start == ["bash", str(root / "command.sh")]


@pytest.mark.parametrize(
    ("gpus", "resources", "refusal"),
    [
        pytest.param("", {}, "names 0 \\(none\\)", id="none-named"),
        pytest.param("1", {"gpus_per_node": 2}, "asks for 2 GPUs and HTR_TRAIN_LOCAL_GPUS names 1", id="too-few"),
        pytest.param("1,2", {"nodes": 2, "gpus_per_node": 2}, "asks for 2 nodes", id="two-nodes"),
        pytest.param("1,x", {}, "GPU ids, comma-separated", id="not-an-id"),
        pytest.param("1,1", {"gpus_per_node": 2}, "each once", id="named-twice"),
    ],
)
def test_plan_run_refuses_a_local_run_its_named_gpus_do_not_fit(
    make_set: MakeSet, settings: Settings, gpus: str, resources: dict[str, JsonValue], refusal: str
) -> None:
    with pytest.raises(JobRefused, match=refusal):
        plan_run(_spec(make_set, **resources), Where.LOCAL, settings.model_copy(update={"local_gpus": gpus}))
