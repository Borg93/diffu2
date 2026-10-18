"""Device backend — pick where inference runs and adapt the pipeline to it.

Three targets, auto-detected (override with ``DIFFU_STUDIO_DEVICE=zerogpu|gpu|cpu`` or the CLI flag):

- **zerogpu** — Hugging Face Spaces on-demand GPU. GPU work must be wrapped in ``@spaces.GPU``; that is
  what :func:`gpu_task` does (and it supports *generator* functions, so streaming keeps the GPU).
- **gpu** — a local CUDA GPU (the training box: pin to GPU0, memory-capped, never OOM the training run).
- **cpu** — fallback / Spaces-CPU. Correct but NOT interactive for whole pages (a ~2B-param diffusion
  backbone over N steps can't stream a page at "live" speed), so the UI shows progress, not animation.

Nothing here imports torch at module load — a mismatched checkpoint or a CPU box must still be able to
import the package and read the config.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    import torch


class DeviceKind(StrEnum):
    zerogpu = "zerogpu"
    gpu = "gpu"
    cpu = "cpu"


class Device(BaseModel):
    """Resolved inference target: where tensors live, in what dtype, and whether it can stream live."""

    model_config = ConfigDict(frozen=True)

    kind: DeviceKind
    torch_device: str  # "cuda" | "cpu"
    dtype: str  # a torch dtype name: "float32" | "bfloat16"
    interactive: bool  # fast enough to animate a page line-by-line (else show a progress bar)

    @property
    def label(self) -> str:
        return f"{self.kind.value} · {self.torch_device} · {self.dtype}"


def resolve_device(override: str | None = None) -> Device:
    """Resolve the inference device from an explicit choice, then ``DIFFU_STUDIO_DEVICE``, then autodetect."""
    choice = (override or os.environ.get("DIFFU_STUDIO_DEVICE") or "auto").lower()
    kind = _detect(choice)
    if kind is DeviceKind.cpu:
        return Device(kind=kind, torch_device="cpu", dtype="float32", interactive=False)
    # gpu and zerogpu both run on CUDA. Default to float32 — the proven, correctness-first path; bf16 is an
    # opt-in speed/VRAM lever (DIFFU_STUDIO_DTYPE=bfloat16), kept out of the default so page CER stays put.
    dtype = "bfloat16" if os.environ.get("DIFFU_STUDIO_DTYPE") == "bfloat16" else "float32"
    return Device(kind=kind, torch_device="cuda", dtype=dtype, interactive=True)


def _detect(choice: str) -> DeviceKind:
    if choice in (DeviceKind.zerogpu, DeviceKind.gpu, DeviceKind.cpu):
        return DeviceKind(choice)
    if _on_zerogpu():
        return DeviceKind.zerogpu
    return DeviceKind.gpu if _cuda_available() else DeviceKind.cpu


def _on_zerogpu() -> bool:
    return bool(os.environ.get("SPACES_ZERO_GPU"))


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def torch_dtype(device: Device) -> torch.dtype:
    """The ``torch`` dtype object named by ``device.dtype`` (e.g. ``torch.float32``)."""
    import torch

    return getattr(torch, device.dtype)


def configure_memory() -> None:
    """Cap CUDA fragmentation so the studio coexists with the training run. Must run BEFORE torch is imported
    (the allocator reads this at init), so the CLI calls it as its first line."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def gpu_task[**P, R](
    func: Callable[P, R] | None = None, *, duration: int | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap a GPU entry point so ZeroGPU allocates a GPU while it runs; a no-op off Spaces.

    ``spaces.GPU`` supports generator functions, so a streaming ``stream_line`` / ``stream_page`` keeps the
    GPU allocated across all its yields. Off ZeroGPU (local gpu / cpu) the function is returned unchanged.

    Usable bare (``gpu_task(fn)`` / ``@gpu_task``) or with a ZeroGPU time budget in seconds
    (``gpu_task(fn, duration=90)`` / ``@gpu_task(duration=90)``) for a long call like the PiD upscale.
    """

    def wrap(f: Callable[P, R]) -> Callable[P, R]:
        if not _on_zerogpu():
            return f
        import importlib

        try:  # `spaces` only exists inside a HF Spaces runtime — import by name so it's not a hard dep
            spaces = importlib.import_module("spaces")
        except ImportError:
            return f
        return spaces.GPU(f, duration=duration) if duration is not None else spaces.GPU(f)

    return wrap if func is None else wrap(func)
