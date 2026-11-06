"""CER while ms-swift trains: at each evaluation (and, if asked, once when training starts or resumes) the model being
trained reads fixed samples of lines or regions, greedily, and their error rates are logged beside the eval loss
(``eval_cer_<set>``). It watches the trend only: the scores compared with the baseline come from a checkpoint read
through vLLM (``read``), and the checkpoint kept is still the one of least eval loss.

The sets come from the environment (:data:`CER_SETS_ENV`, a JSON object of name to :class:`CerSet`), since ms-swift
passes a callback no arguments of its own; ``swift_plugin.py`` beside this module registers the callback."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from diffu_htr_train.evaluate.cer import score, trocr_normalize, whitespace
from diffu_htr_train.evaluate.lines import EvalLine

CER_SETS_ENV = "DIFFU_CER_SETS"
AT_START_ENV = "DIFFU_CER_AT_START"
PROMPT = "<image>Text Recognition:"  # as the training samples ask it
_THINKING = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


class CerSet(BaseModel):
    """A sample to read at each evaluation: its lines (``set-lines``' format), the tokens a reading may take (a line
    512, a region as OpenBMB's SDK allows a text block, 4096) and how many are read at once on a GPU."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: Path
    max_tokens: int = Field(default=512, gt=0)
    batch: int = Field(default=16, gt=0)


CER_SETS = TypeAdapter(dict[str, CerSet])


class Reading(BaseModel):
    """One line's reading: the answer without its (empty) thinking block, and why generation ended."""

    model_config = ConfigDict(frozen=True)

    prediction: str
    finished: str | None


def cer_sets(environ: dict[str, str] | None = None) -> dict[str, CerSet]:
    """The sets the environment names (none when unset)."""
    raw = (os.environ if environ is None else environ).get(CER_SETS_ENV)
    return CER_SETS.validate_json(raw) if raw else {}


def at_start(environ: dict[str, str] | None = None) -> bool:
    """Whether to read the sets once when training starts or resumes too (the checkpoint's own score, and a check
    of the callback within minutes rather than at the first evaluation)."""
    return (os.environ if environ is None else environ).get(AT_START_ENV, "") in {"1", "true", "yes"}


def answer(content: str | None) -> str:
    """A reply's text, its leading thinking block (empty for a model trained with the non-thinking prefix) removed."""
    return _THINKING.sub("", content or "", count=1).strip()


def metrics(name: str, lines: Sequence[EvalLine], readings: Sequence[Reading]) -> dict[str, float]:
    """``name``'s rates, as ``cer`` scores a predictions file: the per-line mean CER (whitespace normalised), the
    corpus CER, the per-line mean with the baseline's normalisation on both sides, and the readings cut off."""
    pairs = [(line.text, reading.prediction) for line, reading in zip(lines, readings, strict=True)]
    raw, normalised = score(pairs, whitespace), score(pairs, trocr_normalize)
    return {
        f"cer_{name}": raw.mean_cer,
        f"cer_corpus_{name}": raw.corpus_cer,
        f"cer_norm_{name}": normalised.mean_cer,
        f"cut_{name}": float(sum(reading.finished == "length" for reading in readings)),
    }


def write_readings(path: Path, lines: Sequence[EvalLine], readings: Sequence[Reading]) -> None:
    """The readings as ``read`` writes predictions, so ``cer`` scores them again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for line, reading in zip(lines, readings, strict=True):
            row = {**line.model_dump(), "prediction": reading.prediction, "finished": reading.finished}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


class Recogniser(Protocol):
    """What reads a batch of lines: ms-swift's engine over the model in training, or one loaded from a checkpoint."""

    def __call__(self, lines: Sequence[EvalLine], cer_set: CerSet) -> list[Reading]: ...
