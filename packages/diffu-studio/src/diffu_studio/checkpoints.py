"""Checkpoint discovery + loading, with the architecture guard wired in.

The studio must never silently generate gibberish from a checkpoint whose text-conditioner doesn't fit
the code (the failure that cost a whole debugging session). :func:`load_model` therefore runs
``diffu.generate.assert_conditioner_loaded`` right after loading and re-raises any mismatch as
:class:`CheckpointMismatch`, which the UI/API turn into a clear message instead of confident nonsense.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from diffu_studio.device import Device, torch_dtype

if TYPE_CHECKING:
    from diffu.config import Config
    from diffu.model import Diffu


class ArchConfig(BaseModel):
    """The architecture flags a checkpoint was TRAINED with — must match or the conditioner loads wrong.

    Defaults track the current best run (``exp_sd35_fast``: ``--glyph-line --no-style-tokens``). ``qk_norm``
    and ``use_unifont`` are NOT here: they are auto-detected per checkpoint from its weights at load time.
    """

    model_config = ConfigDict(frozen=True)

    glyph_line: bool = True
    style_in_context: bool = False
    glyph_concat: bool = False
    line_height: int = 64  # crop height the checkpoint was trained at (64px runs; 128 for exp_sd35_128)


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


def infer_qk_norm(ckpt: str) -> str | None:
    """``"rms_norm"`` if the checkpoint was trained with SD3.5 qk-norm (has ``...attn.norm_q.*``), else None."""
    try:
        from safetensors import safe_open

        with safe_open(ckpt, framework="pt") as handle:
            return "rms_norm" if any(".norm_q." in k for k in handle.keys()) else None  # noqa: SIM118
    except Exception:  # noqa: BLE001 — a .pt file or unreadable header: fall back to no qk-norm
        return None


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
    """Build Diffu with ``arch`` (+ auto-detected qk-norm/unifont), load ``path``, and verify the conditioner.

    An optional Stage-3 ``lora`` adapter (dir or Hub id) is applied on top of the base checkpoint at
    ``lora_scale`` — after the conditioner guard, before any compile (mirrors ``DiffuPipeline``).
    Raises :class:`CheckpointMismatch` if the checkpoint's text-conditioner doesn't fit the code.
    """
    from diffu.config import Config
    from diffu.generate import assert_conditioner_loaded, load_checkpoint
    from diffu.model import Diffu

    weights = resolve_weights(path)
    cfg = Config()
    cfg.data.line_height = arch.line_height  # config.py default may differ from the checkpoint's train res
    cfg.cond.glyph_line = arch.glyph_line
    cfg.cond.style_in_context = arch.style_in_context
    cfg.cond.glyph_concat = arch.glyph_concat
    qk = infer_qk_norm(weights)  # the content-stencil font is keyed to the same signal (see app-era note)
    cfg.backbone.qk_norm = qk
    cfg.cond.use_unifont = qk is not None

    model = Diffu(cfg).to(device.torch_device, dtype=torch_dtype(device)).eval()
    load_checkpoint(model, weights)
    try:
        assert_conditioner_loaded(model, weights)
    except RuntimeError as exc:
        raise CheckpointMismatch(str(exc)) from exc
    if lora:  # a Stage-3 LoRA adapter on top of the base checkpoint (guarded + verified first)
        from diffu.lora import load_lora

        load_lora(model, lora, scale=lora_scale)
    return LoadedModel(model=model, cfg=cfg, path=weights, tag=f"SD3.{'5' if qk else '0'}")
