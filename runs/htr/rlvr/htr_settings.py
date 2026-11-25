"""The settings the task's hooks share with the trainer, read from the one training YAML.

The recipe hands the training YAML to NeMo RL only; the task's hooks (candidates, selection, evaluation) get no
configuration. So that a weight or a temperature is set in one place, the hooks read the same file: the path in
``HTR_RLVR_CONFIG`` (the launcher sets it to the ``training_config`` it passes on), ``grpo_htr.yaml`` beside this
module when unset. A stage-level ``training.*`` override on the command line is NOT seen here; change the YAML (or a
copy that has ``defaults: grpo_htr.yaml``) instead.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

from htr_candidates import rules

HERE = Path(__file__).resolve().parent
DATA = Path("/mnt/scratch/gabriel/htr/rlvr/data")


def config_path() -> Path:
    """The training YAML in force."""
    return Path(os.environ.get("HTR_RLVR_CONFIG") or HERE / "grpo_htr.yaml").resolve()


def data_dir() -> Path:
    """The dataset bundle the candidates are drawn from (``HTR_RLVR_DATA``; the launcher sets it to ``data_dir``)."""
    return Path(os.environ.get("HTR_RLVR_DATA") or DATA).resolve()


def _load(path: Path) -> Any:
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    parents = config.pop("defaults", None)
    if parents is None:
        return config
    merged = OmegaConf.create({})
    for parent in [parents] if isinstance(parents, str) else list(parents):
        parent_path = Path(parent) if str(parent).startswith("/") else path.parent / parent
        merged = OmegaConf.merge(merged, _load(parent_path))
    return OmegaConf.merge(merged, config)


@cache
def settings() -> dict[str, Any]:
    """The values the hooks need, as plain Python."""
    from omegaconf import OmegaConf

    config = _load(config_path())

    def get(key: str) -> Any:
        value = OmegaConf.select(config, key, throw_on_missing=True)
        if value is None:
            raise KeyError(f"{key} is not set in {config_path()}")
        return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value

    candidates = OmegaConf.select(config, "htr.candidates", default=None)  # unset in grpo_htr.yaml: pilot 1's draw
    return {
        "config": str(config_path()),
        "candidates": rules(OmegaConf.to_container(candidates, resolve=True) if candidates is not None else None),
        "weights": {name: float(weight) for name, weight in get("htr.weights").items()},
        "loop": {name: int(value) for name, value in get("htr.loop").items()},
        "greedy_repetition_penalties": [float(value) for value in get("htr.greedy_repetition_penalties")],
        "failing_share": float(get("htr.failing_share")),
        "informed_share": float(get("htr.informed_share")),
        "temperature": float(get("policy.generation.temperature")),
        "top_p": float(get("policy.generation.top_p")),
        "max_new_tokens": int(get("policy.generation.max_new_tokens")),
        "max_model_len": int(get("policy.max_total_sequence_length")),
        "seed": int(get("grpo.seed")),
        "max_num_batched_tokens": int(get("policy.generation.vllm_kwargs.max_num_batched_tokens")),
        "kv_cache_memory_bytes": int(get("policy.generation.vllm_kwargs.kv_cache_memory_bytes")),
    }
