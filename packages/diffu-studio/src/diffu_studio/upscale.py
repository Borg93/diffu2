"""PiD 4× super-resolution for the studio — a thin re-export of the shared ``diffu_page.upscale`` core.

The implementation lives in ONE place (``diffu_page.upscale``) so the interactive studio and the Ray page
pipeline can never drift. This module exists only so studio call sites keep importing from
``diffu_studio.upscale``; it adds nothing but the re-export.

Import-safe: ``diffu_page.upscale`` touches no torch/pid/CUDA at module load, so importing this is free on a
CPU box or a machine without PiD. The GPU work happens lazily inside :func:`load_pid` / :func:`pid_upscale`.
"""

from __future__ import annotations

from diffu_page.upscale import (
    load_pid,
    pid_available,
    pid_upscale,
    prefetch_weights,
)

__all__ = ["load_pid", "pid_available", "pid_upscale", "prefetch_weights"]
