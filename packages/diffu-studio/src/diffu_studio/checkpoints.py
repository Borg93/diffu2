"""Checkpoint discovery + loading, with the architecture guard wired in.

The studio must never silently generate gibberish from a checkpoint whose text-conditioner doesn't fit
the code (the failure that cost a whole debugging session). :func:`load_model` therefore loads through
``diffu.generate.build_model`` (which runs ``assert_conditioner_loaded`` right after the weights) and
re-raises any mismatch as :class:`CheckpointMismatch`, which the UI/API turn into a clear message instead
of confident nonsense.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from diffu.config import DataConfig
from diffu_studio.device import Device, torch_dtype

if TYPE_CHECKING:
    from diffu.config import Config
    from diffu.model import Diffu


class ArchConfig(BaseModel):
    """The architecture knobs for a checkpoint WITHOUT a ``config.json`` (runs that predate it).

    diffu-train saves the full model config as ``config.json`` next to the weights; when it is there, it is
    the architecture and these are ignored.
    """

    model_config = ConfigDict(frozen=True)

    line_height: int = (
        DataConfig().line_height
    )  # crop height the checkpoint was trained at (old 64px runs: 64)


class Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str  # "<run>/<step>"
    path: str
    step: int


class LoadedModel(BaseModel):
    """A built + loaded model ready to sample, plus the config it was built with and a short arch tag."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model: Diffu
    cfg: Config
    path: str
    tag: str  # "SD3.5" / "SD3.0"


class CheckpointMismatch(RuntimeError):
    """A checkpoint's architecture doesn't fit this code — surfaced to the user, never silently ignored."""


def discover_checkpoints(root: str = "checkpoints") -> list[Checkpoint]:
    """All ``<run>/step_<N>`` (and ``final``) checkpoints under ``root``, MOST-RECENTLY-TRAINED first, so the
    default is the active run's latest — not a stale run that merely has a higher step count. Prefers the EMA
    export (what sampling used) over the raw weights."""
    base = Path(root)
    found: list[Checkpoint] = []
    for step_dir in base.glob("*/step_*"):
        tail = step_dir.name.partition("_")[2]
        path = _prefer_ema(step_dir)
        if tail.isdigit() and path:
            found.append(
                Checkpoint(label=f"{step_dir.parent.name}/{step_dir.name}", path=path, step=int(tail))
            )
    for final_dir in base.glob("*/final"):
        path = _prefer_ema(final_dir)
        if path:
            found.append(Checkpoint(label=f"{final_dir.parent.name}/final", path=path, step=10**12))
    found.sort(key=lambda c: Path(c.path).stat().st_mtime, reverse=True)  # newest-trained first (across runs)
    return found


def latest_checkpoint(root: str = "checkpoints") -> Checkpoint | None:
    """The newest checkpoint under ``root`` (the default the studio boots with), or None if there are none."""
    found = discover_checkpoints(root)
    return found[0] if found else None


def _prefer_ema(step_dir: Path) -> str | None:
    for name in ("ema_model.safetensors", "model.safetensors"):
        if (step_dir / name).exists():
            return str(step_dir / name)
    return None


def resolve_weights(path: str) -> str:
    """Accept a ``step_*`` DIR or a ``.safetensors`` file → the weights file (EMA preferred for a dir)."""
    p = Path(path)
    if p.is_dir():
        return _prefer_ema(p) or str(p / "model.safetensors")
    return path


def load_model(
    path: str,
    arch: ArchConfig,
    device: Device,
    *,
    lora: str | None = None,
    lora_scale: float = 1.0,
) -> LoadedModel:
    """Build Diffu from the checkpoint's ``config.json`` (else ``arch``), load ``path``, verify the conditioner.

    Loads through ``diffu.generate.build_model``: an optional Stage-3 ``lora`` adapter (dir or Hub id) is
    applied on top of the base checkpoint at ``lora_scale`` — after the conditioner guard, before any compile
    (mirrors ``DiffuPipeline``). Raises :class:`CheckpointMismatch` if the checkpoint doesn't fit the code.
    """
    import torch

    # Config/Diffu are LoadedModel's field types: pydantic resolves its deferred annotations from the frame
    # that first instantiates it — this one — so they must be importable names here.
    from diffu.config import Config  # noqa: F401
    from diffu.generate import build_model, load_config
    from diffu.model import Diffu  # noqa: F401

    weights = resolve_weights(path)
    cfg = load_config(weights)
    if not (Path(weights).parent / "config.json").is_file():  # an older run: no saved config
        cfg.data.line_height = arch.line_height  # config.py's default may differ from its train res
    try:
        model = build_model(
            cfg,
            weights,
            device=device.torch_device,
            dtype=torch_dtype(device),
            lora=lora,
            lora_scale=lora_scale,
        )
    except torch.OutOfMemoryError:
        raise  # a full GPU, not a mismatch
    except RuntimeError as exc:  # build_model's arch guards: input channels, then the text-conditioner
        raise CheckpointMismatch(str(exc)) from exc
    tag = f"SD3.{'5' if model.cfg.backbone.qk_norm else '0'}"
    return LoadedModel(model=model, cfg=model.cfg, path=weights, tag=tag)
