"""diffu-runner — run Diffu at scale to forge synthetic HTR datasets.

A Ray-Data pipeline of two stateful actors (generate → CER-gate) that turns the single-line model into a
million-line, leakage-safe, quality-gated training set. Heavy/optional Ray dep isolated in this workspace
member. See :func:`diffu_runner.pipeline.run`.
"""

from __future__ import annotations

from .gate import GateConfig, cer_ok, ink_fraction, ink_ok
from .pages import PageConfig, coherent_docs, run_pages, varied_job
from .pipeline import RunConfig, build_worklist, run, run_target

__all__ = [
    "GateConfig",
    "PageConfig",
    "RunConfig",
    "build_worklist",
    "cer_ok",
    "coherent_docs",
    "ink_fraction",
    "ink_ok",
    "run",
    "run_pages",
    "run_target",
    "varied_job",
]
