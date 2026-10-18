"""The frontend-agnostic core API — no gradio, no FastAPI.

Everything the UI (``ui.py``) and the server (``server.py``) need to generate lines and pages, discover
checkpoints, list templates, and read the default configs. A different frontend (custom HTML/JS, a
notebook, a batch script) can import exactly this and nothing UI-specific.
"""

from __future__ import annotations

from diffu_studio.checkpoints import (
    ArchConfig,
    Checkpoint,
    CheckpointMismatch,
    LoadedModel,
    discover_checkpoints,
    latest_checkpoint,
    load_model,
)
from diffu_studio.configs import (
    SampleSettings,
    build_augment,
    build_gen_config,
    default_configs,
    list_formats,
    list_templates,
)
from diffu_studio.device import Device, DeviceKind, gpu_task, resolve_device
from diffu_studio.engine import StudioEngine
from diffu_studio.line import Recognizer, sample_gallery, stream_line, style_from_pil
from diffu_studio.page import PageRequest, stream_page
from diffu_studio.steps import LineStep, PageStep


def list_checkpoints(root: str = "checkpoints") -> list[Checkpoint]:
    """All discoverable checkpoints under ``root``, newest step first (the API/UI name for discovery)."""
    return discover_checkpoints(root)


__all__ = [
    "ArchConfig",
    "Checkpoint",
    "CheckpointMismatch",
    "Device",
    "DeviceKind",
    "LineStep",
    "LoadedModel",
    "PageRequest",
    "PageStep",
    "Recognizer",
    "SampleSettings",
    "StudioEngine",
    "build_augment",
    "build_gen_config",
    "default_configs",
    "gpu_task",
    "latest_checkpoint",
    "list_checkpoints",
    "list_formats",
    "list_templates",
    "load_model",
    "resolve_device",
    "sample_gallery",
    "stream_line",
    "stream_page",
    "style_from_pil",
]
