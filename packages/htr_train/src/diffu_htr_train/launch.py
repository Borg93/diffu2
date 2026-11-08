"""A run planned (:func:`plan_run`: its spec checked, its commands rendered, its scripts written out as text) and then
written and started where it runs (:data:`LAUNCHERS`): on the Spark under its memory guard, on a Slurm cluster by
sbatch, or on a server's own GPUs by bash. Every run's folder ``<runs>/<name>/`` holds the validated spec (``job.yaml``), what it was run from
(``manifest.json``) and its scripts: ``steps.sh`` (its commands, in order) and the launcher's (``command.sh``, the
guard's call or the steps' on a server, or ``job.sbatch``).
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ConfigDict, JsonValue

from diffu_htr_train.jobs import DSparkJob, JobRefused, RunSpec, ToolEnv, Where, check_run, check_where
from diffu_htr_train.records import meta_path
from diffu_htr_train.render import Command, Flag, RunPaths, flags, render
from diffu_htr_train.sets import SetRefused, read_options
from diffu_htr_train.settings import Settings

READY_SECONDS = 1800  # how long a served model may take to answer its health check before the run gives up
MASTER_PORT = 29500

# What each launcher's steps set before the commands: the Spark's one GPU and capped CPU threads (a training shares
# the box); on Slurm, ms-swift's multi-node variables from the task's own (one srun task a node), the models staged;
# on a server, its named GPUs (:func:`preamble`) and capped CPU threads (ms-swift starts a process a GPU)
PREAMBLES: dict[Where, list[str]] = {
    Where.SPARK: [
        "export CUDA_VISIBLE_DEVICES=0",
        "export OMP_NUM_THREADS=4",
        'export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"',
    ],
    Where.SLURM: [
        "export NNODES=$SLURM_NNODES NODE_RANK=$SLURM_NODEID",
        'export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"',
        f"export MASTER_PORT={MASTER_PORT}",
        "export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8",
    ],
    Where.LOCAL: [
        "export OMP_NUM_THREADS=8",
        'export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"',
    ],
}


def preamble(where: Where, settings: Settings) -> list[str]:
    """What the steps set before the commands (:data:`PREAMBLES`); a local run's GPUs first, as the settings name
    them."""
    if where is Where.LOCAL:
        gpus = ",".join(map(str, settings.local_gpu_ids()))
        return [f"export CUDA_VISIBLE_DEVICES={gpus}", *PREAMBLES[where]]
    return PREAMBLES[where]


class Plan(BaseModel):
    """A run ready to write and start: its spec, where it runs, its paths, its commands and its scripts' text (each
    by its file name in the run's folder)."""

    model_config = ConfigDict(frozen=True)

    spec: RunSpec
    where: Where
    paths: RunPaths
    commands: list[Command]
    scripts: dict[str, str]
    start: list[str]


def shell(command: Command) -> str:
    """A command as one line of bash: its environment, then its argv, each word quoted where it must be."""
    environ = [f"{key}={shlex.quote(given)}" for key, given in sorted(command.environ.items())]
    return shlex.join(command.argv) if not environ else f"env {' '.join(environ)} {shlex.join(command.argv)}"


def _serve_lines(command: Command, *, ready: str, log: Path) -> list[str]:
    """Start a served model in the background, stopped when the steps end, and wait until it answers at ``ready`` (at
    most :data:`READY_SECONDS`), failing if it dies first."""
    return [
        f"{shell(command)} > {shlex.quote(str(log))} 2>&1 &",
        "SERVED=$!",
        'trap \'kill "$SERVED" 2>/dev/null; wait "$SERVED" 2>/dev/null || true\' EXIT',
        f"for _ in $(seq {READY_SECONDS // 5}); do",
        f"    curl -sf {shlex.quote(ready)} > /dev/null && break",
        '    kill -0 "$SERVED" 2>/dev/null || { echo "the served model stopped: see its log" >&2; exit 1; }',
        "    sleep 5",
        "done",
        f"curl -sf {shlex.quote(ready)} > /dev/null",
    ]


def _resume_lines(output: Path) -> list[str]:
    """``RESUME`` set to ms-swift's flag for the newest whole checkpoint in ``output`` (one holding its
    ``trainer_state.json``, which the trainer writes last: a crash while saving leaves a checkpoint without it), or
    left empty. The trainer then skips the batches the checkpoint's steps trained; ``--ignore_data_skip`` is never
    passed, since it restarts the epoch's pass at its first batch and trains those batches again."""
    newest = f"ls -d {shlex.quote(str(output))}/checkpoint-*/trainer_state.json 2>/dev/null | sort -V | tail -n 1"
    return [
        "RESUME=()",
        f"CHECKPOINT=$({newest} || true)",
        'if [ -n "$CHECKPOINT" ]; then',
        '    RESUME=(--resume_from_checkpoint "${CHECKPOINT%/trainer_state.json}")',
        '    echo "htr-train: resuming from ${RESUME[1]}" >&2',
        "fi",
    ]


def steps_script(
    spec: RunSpec, *, where: Where, commands: list[Command], paths: RunPaths, setup: list[str] | None = None
) -> str:
    """The run's commands as a bash script, in order, after its launcher's preamble (``setup``; None:
    :data:`PREAMBLES`'s); a served model started in the background and waited for, a training resumed from its newest
    whole checkpoint when the steps run again (:func:`_resume_lines`)."""
    lines = [
        "#!/bin/bash",
        f"# htr-train run {spec.name} ({spec.job.kind}), written by htr-train launch --where {where}.",
        "set -euo pipefail",
        *(PREAMBLES[where] if setup is None else setup),
    ]
    for command in commands:
        match command:
            case Command(role="serve", ready=str() as ready):
                lines += _serve_lines(command, ready=ready, log=paths.root / "serve.log")
            case Command(resume_from=Path() as output):
                lines += [*_resume_lines(output), f'{shell(command)} "${{RESUME[@]}}"']
            case _:
                lines.append(shell(command))
    return "\n".join(lines) + "\n"


def spark_script(spec: RunSpec, paths: RunPaths, *, guard: Path, floor_gb: int) -> str:
    """The call of the memory guard at ``guard`` that runs the steps on the Spark: one guarded GPU job at a time,
    started only with ``need_gb`` plus the guard's floor (``floor_gb``) free, killed if free memory falls below the
    floor."""
    call = [
        str(guard.resolve()),
        "--need-gb",
        str(spec.resources.need_gb),
        "--floor-gb",
        str(floor_gb),
        "--name",
        f"htr-{spec.name}",
        "--log",
        str((paths.root / "guard.log").resolve()),
        "--",
        "bash",
        str((paths.root / "steps.sh").resolve()),
    ]
    return "\n".join(["#!/bin/bash", "set -euo pipefail", f"exec {shlex.join(call)}"]) + "\n"


def _hours(hours: float) -> str:
    """Slurm's ``--time``, in whole minutes."""
    minutes = round(hours * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def slurm_script(spec: RunSpec, paths: RunPaths) -> str:
    """The sbatch file that runs the steps on a Slurm cluster, one srun task a node (ms-swift starts a process a GPU
    there); the partition and account are the site's, given to sbatch."""
    resources = spec.resources
    header = [
        f"--job-name=htr-{spec.name}",
        f"--nodes={resources.nodes}",
        "--ntasks-per-node=1",
        f"--gpus-per-node={resources.gpus_per_node}",
        f"--cpus-per-task={resources.cpus_per_task}",
        "--mem=0",
        f"--time={_hours(resources.hours)}",
        f"--output={(paths.root / 'slurm-%j.out').resolve()}",
    ]
    return (
        "\n".join(
            [
                "#!/bin/bash",
                *(f"#SBATCH {line}" for line in header),
                "set -euo pipefail",
                f"srun bash {shlex.quote(str((paths.root / 'steps.sh').resolve()))}",
            ]
        )
        + "\n"
    )


class Launch(BaseModel):
    """A launcher's script for a run: its file name in the run's folder, its text, and the command that starts it."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str
    start: list[str]


class Launcher(Protocol):
    """A launcher's script for a run, from the machine's settings and the guard's floor :func:`plan_run` read."""

    def __call__(self, spec: RunSpec, paths: RunPaths, *, settings: Settings, floor_gb: int) -> Launch: ...


def _spark(spec: RunSpec, paths: RunPaths, *, settings: Settings, floor_gb: int) -> Launch:
    """The guard's call, run with bash."""
    script = paths.root / "command.sh"
    text = spark_script(spec, paths, guard=settings.guard, floor_gb=floor_gb)
    return Launch(name=script.name, text=text, start=["bash", str(script)])


def _slurm(spec: RunSpec, paths: RunPaths, *, settings: Settings, floor_gb: int) -> Launch:
    """The sbatch file, submitted with the site's partition and account where they are set (an empty one is unset);
    the guard's floor is the Spark's only."""
    script = paths.root / "job.sbatch"
    site: list[Flag] = [
        ("--partition", settings.slurm_partition or None),
        ("--account", settings.slurm_account or None),
    ]
    return Launch(name=script.name, text=slurm_script(spec, paths), start=["sbatch", *flags(site), str(script)])


def _local(spec: RunSpec, paths: RunPaths, *, settings: Settings, floor_gb: int) -> Launch:
    """The steps, run with bash on the server's own GPUs (their preamble names them); no guard: the GPUs are the
    run's."""
    script = paths.root / "command.sh"
    text = "\n".join(
        ["#!/bin/bash", "set -euo pipefail", f"exec bash {shlex.quote(str((paths.root / 'steps.sh').resolve()))}"]
    )
    return Launch(name=script.name, text=text + "\n", start=["bash", str(script)])


LAUNCHERS: dict[Where, Launcher] = {Where.SPARK: _spark, Where.SLURM: _slurm, Where.LOCAL: _local}


def plan_run(spec: RunSpec, where: Where, settings: Settings) -> Plan:
    """The run ``spec`` launched ``where``: checked (:func:`diffu_htr_train.jobs.check_run`,
    :func:`diffu_htr_train.jobs.check_where`), rendered, its scripts written out as text; nothing is written or
    started.

    Raises:
        JobRefused: If the run fails a check.
    """
    check_run(spec)
    try:
        floor = settings.guard_floor()
        local_gpus = settings.local_gpu_ids()
    except ValueError as error:
        raise JobRefused(str(error)) from error
    check_where(spec, where, memory_gb=settings.memory_gb, floor_gb=floor, local_gpus=local_gpus)
    paths = RunPaths(
        root=settings.runs.resolve() / spec.name, envs=settings.envs.resolve(), trackio_project=settings.trackio_project
    )
    commands = render(spec, paths)
    launch = LAUNCHERS[where](spec, paths, settings=settings, floor_gb=floor)
    steps = steps_script(spec, where=where, commands=commands, paths=paths, setup=preamble(where, settings))
    scripts = {"steps.sh": steps, launch.name: launch.text}
    return Plan(spec=spec, where=where, paths=paths, commands=commands, scripts=scripts, start=launch.start)


class DataFile(BaseModel):
    """A file a run reads: its path, digest and number of lines, and for a set, how it was built (its
    ``<stem>.options.json``, as written; None where it records none)."""

    model_config = ConfigDict(frozen=True)

    path: str
    sha256: str
    lines: int
    options: JsonValue = None


class Manifest(BaseModel):
    """What a run was started from (``manifest.json``): the repository's commit and whether it had changes, each tool
    environment's lock digest (None: no lock), the models, the sets and the exact commands."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    kind: str
    where: Where
    created: str
    git_commit: str
    git_dirty: bool
    env_locks: dict[ToolEnv, str | None]
    models: list[str]
    data: list[DataFile]
    commands: list[Command]


def _sha256(path: Path) -> str:
    return _digest_and_lines(path)[0]


def _digest_and_lines(path: Path) -> tuple[str, int]:
    """The file's sha256 and its number of lines, in one read."""
    digest = hashlib.sha256()
    lines = 0
    last = b"\n"
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
            lines += block.count(b"\n")
            last = block[-1:]
    return digest.hexdigest(), lines + (last != b"\n")


def _git(*args: str) -> str:
    """A git command's output in the repository htr-train is run from.

    Raises:
        JobRefused: Outside a git repository: a run's manifest names the commit it ran from.
    """
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise JobRefused(f"git {' '.join(args)} failed ({result.stderr.strip()}): launch runs from the repository")
    return result.stdout.strip()


def _data_files(spec: RunSpec) -> list[Path]:
    """The sets a run reads, its meta files beside them."""
    if isinstance(spec.job, DSparkJob):
        return [spec.job.responses]
    if spec.data is None:
        return []
    sets = [part.path for part in spec.data.train] + ([spec.data.dev] if spec.data.dev else [])
    return [file for path in sets for file in (path, meta_path(path))]


def _data_file(path: Path) -> DataFile:
    """A file a run reads, with its set's options where they are recorded.

    Raises:
        JobRefused: If the set's record of its options is not JSON.
    """
    try:
        options = read_options(path)
    except SetRefused as error:
        raise JobRefused(str(error)) from error
    sha256, lines = _digest_and_lines(path)
    return DataFile(path=str(path.resolve()), sha256=sha256, lines=lines, options=options)


def manifest(plan: Plan) -> Manifest:
    """The manifest of a planned run, now."""
    job = plan.spec.job
    model = job.target if isinstance(job, DSparkJob) else job.model
    envs = {command.env for command in plan.commands}
    return Manifest(
        name=plan.spec.name,
        description=plan.spec.description,
        kind=job.kind,
        where=plan.where,
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        env_locks={
            env: _sha256(lock) if (lock := plan.paths.envs / env / "uv.lock").is_file() else None
            for env in sorted(envs)
        },
        models=[f"{model.repo}@{model.revision}" if model.revision else model.repo],
        data=[_data_file(path) for path in _data_files(plan.spec)],
        commands=plan.commands,
    )


def write_run(plan: Plan) -> Path:
    """The run's folder written: its spec, manifest and scripts (executable); the folder.

    Raises:
        JobRefused: If the run's folder already holds a run (ms-swift writes its checkpoints straight into
            ``<run>/output``, so a second launch would mix with the first), or outside a git repository (:func:`_git`).
    """
    root = plan.paths.root
    if (root / "manifest.json").exists():
        raise JobRefused(f"{root} holds a run already: give the spec another name, or move that folder away")
    record = manifest(plan)
    root.mkdir(parents=True, exist_ok=True)
    (root / "job.yaml").write_text(
        yaml.safe_dump(plan.spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (root / "manifest.json").write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    for name, text in plan.scripts.items():
        (root / name).write_text(text, encoding="utf-8")
        (root / name).chmod(0o755)
    return root


def queue_script(plans: list[Plan]) -> str:
    """A queue of Spark runs, one after another, stopping at the first that fails; a run the guard does not start
    because another guarded job holds the GPU (its exit 3) is tried again every 5 minutes, 30 times."""
    lines = [
        "#!/bin/bash",
        "# htr-train runs, one guarded GPU job after another (htr-train smoke).",
        "set -euo pipefail",
        "guarded() {",
        "    local status",
        "    for try in $(seq 30); do",
        '        status=0; "$@" || status=$?',
        '        [ "$status" -ne 3 ] && return "$status"',
        '        echo "the GPU is busy, try $try of 30" >&2; sleep 300',
        "    done",
        "    return 3",
        "}",
    ]
    lines += [f"guarded {shlex.join(plan.start)}" for plan in plans]
    return "\n".join(lines) + "\n"
