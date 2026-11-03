"""The smoke runs on the Spark's GPU, each through the memory guard, one at a time (the ``gpu`` lane, never the
default one; their sets built first, packages/htr_train/README.md): each ends well and writes its last step's
checkpoint, and its loss is finite and lower at the end than at the start: the eval loss where the run evaluates, else
the training loss. It walks the shipped smoke specs on purpose (the lane is "every smoke run"), against the testing
reference's rule of parametrising over inputs to a unit."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from diffu_htr_train.jobs import SwiftSFT, Where, load_run_spec
from diffu_htr_train.launch import plan_run, write_run
from diffu_htr_train.settings import Settings

pytestmark = pytest.mark.gpu

SMOKE = sorted((Path(__file__).parents[1] / "configs" / "smoke").glob("*.yaml"))


def _losses(logging: Path) -> list[float]:
    """ms-swift's logged losses: the eval loss of each evaluation, or the training loss of each step without one."""
    rows = [json.loads(line) for line in logging.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluated = [row["eval_loss"] for row in rows if "eval_loss" in row]
    return evaluated or [row["loss"] for row in rows if "loss" in row]


@pytest.mark.parametrize("spec_path", SMOKE, ids=lambda path: path.stem)
def test_smoke_run_trains_and_saves_with_a_falling_loss(spec_path: Path) -> None:
    spec = load_run_spec(spec_path)
    job = spec.job
    assert isinstance(job, SwiftSFT) and job.max_steps is not None, "a smoke run is an ms-swift run of max_steps"
    plan = plan_run(spec, Where.SPARK, Settings())
    write_run(plan)

    status = subprocess.run(plan.start, check=False).returncode

    assert status == 0, f"see {plan.paths.root / 'guard.log'}"
    output = plan.paths.output
    assert (output / f"checkpoint-{job.max_steps}").is_dir()
    losses = _losses(output / "logging.jsonl")
    assert len(losses) >= 2, f"{len(losses)} losses logged: nothing to compare"
    assert all(math.isfinite(loss) for loss in losses) and losses[-1] < losses[0]
