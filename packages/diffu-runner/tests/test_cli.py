"""CLI seam tests — argv actually reaches RunConfig (the ``cli:app`` entrypoint regression) and paths
arrive ABSOLUTE (Ray workers run in the runtime-env package dir, so a relative ckpt FileNotFounds there)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from diffu_runner import cli
from typer.testing import CliRunner


def _invoke(monkeypatch, extra_args: list[str]):
    """Run the CLI with stubbed pipeline entrypoints; return (result, captured calls)."""
    calls: dict[str, Any] = {}  # Any: tests poke RunConfig attrs on the captured values

    def fake_run(cfg):
        calls["run"] = cfg
        return "out"

    def fake_run_target(cfg, target):
        calls["run_target"] = (cfg, target)
        return "out", {"kept": 0}

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "run_target", fake_run_target)
    args = [
        "--ckpt",
        "ckpt/model.safetensors",
        "--train-jsonl",
        "data/train.jsonl",
        "--out",
        "shards",
        *extra_args,
    ]
    return CliRunner().invoke(cli.app, args), calls


def test_cli_absolutizes_paths_for_ray_workers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result, calls = _invoke(monkeypatch, ["--n", "5"])
    assert result.exit_code == 0, result.output
    cfg = calls["run"]
    assert Path(cfg.ckpt).is_absolute() and cfg.ckpt.endswith("ckpt/model.safetensors")
    assert Path(cfg.train_jsonl).is_absolute()
    assert Path(cfg.out).is_absolute()
    assert cfg.n == 5, "argv must reach RunConfig (the cli:main entrypoint never parsed argv)"


def test_cli_coherent_flag_threads_to_runconfig(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result, calls = _invoke(monkeypatch, ["--n", "3", "--coherent"])
    assert result.exit_code == 0, result.output
    assert calls["run"].coherent is True


def test_cli_defaults_to_the_random_worklist(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result, calls = _invoke(monkeypatch, ["--n", "3"])
    assert result.exit_code == 0, result.output
    assert calls["run"].coherent is False, "--random is the default; coherent banking is opt-in"


def test_cli_threads_lora_to_runconfig(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "adapter").mkdir()  # a LOCAL adapter dir must arrive ABSOLUTE (Ray workers, like ckpt)
    result, calls = _invoke(monkeypatch, ["--n", "3", "--lora", "adapter", "--lora-scale", "0.7"])
    assert result.exit_code == 0, result.output
    cfg = calls["run"]
    assert Path(cfg.lora).is_absolute() and cfg.lora.endswith("adapter")
    assert cfg.lora_scale == 0.7, "adapter strength must thread through"


def test_cli_lora_hub_id_passthrough(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # a Hub id (not a local path) passes through unchanged, never path-mangled
    result, calls = _invoke(monkeypatch, ["--n", "3", "--lora", "Gabriel/diffu-iam-lora"])
    assert result.exit_code == 0, result.output
    assert calls["run"].lora == "Gabriel/diffu-iam-lora"


def test_cli_defaults_to_no_lora(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result, calls = _invoke(monkeypatch, ["--n", "3"])
    assert result.exit_code == 0, result.output
    assert calls["run"].lora is None, "base checkpoint only unless --lora is given"


def test_cli_exact_target_dispatches_run_target(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result, calls = _invoke(
        monkeypatch, ["--n", "7", "--exact-target", "--no-compile", "--gen-dtype", "float32"]
    )
    assert result.exit_code == 0, result.output
    cfg, target = calls["run_target"]
    assert target == 7 and "run" not in calls
    assert cfg.compile_model is False and cfg.gen_dtype == "float32", "perf knobs must thread through"
