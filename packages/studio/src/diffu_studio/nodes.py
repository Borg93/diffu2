"""The studio's canvas nodes: the page pipeline cut where its data is plain values.

- :func:`plan_node`: a template page, its new text and a style in; the page's crop plans out (``PagePlanBlocks``, CPU).
- :class:`DrawNode`: the plans, a seed and the steps in; every crop drawn by one drawer's crop pipeline, out as a sheet
  of the crops and a zip of the crops with their labels. One per :class:`Drawer`, all fed by the same plan, so the
  models (or LoRAs) draw the same page from the same noise, side by side.

gr.Workflow calls a node with its input ports' values in order and passes media as file dicts.
"""

from __future__ import annotations

import threading
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from diffusers.modular_pipelines import ModularPipeline
from gradio_client import utils as client_utils
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, field_validator

from diffu_page.blocks import DrawnCrop, crop_identity
from diffu_page.captions.prompt import PageStyle
from diffu_page.layout import load_xml_layout, running_text
from diffu_page.output import JPEG_QUALITY, jpeg_bytes, plain_file_name
from diffu_page.pipeline import DrawSettings, PagePlanBlocks, crop_blocks, draw_crop, load_pipeline
from diffu_page.plan import CropPlan
from diffu_page.readback import pixel_box
from diffu_studio.graph import draw_fn

SHEET_HEIGHT = 512  # px: every crop of a sheet is scaled to it
SHEET_GAP = 8  # px between the crops of a sheet
_FILE_URL = "/gradio_api/file="


class Drawer(BaseModel):
    """One draw node: a model directory and the LoRAs (NeMo ``model/`` dirs by adapter name) active when it draws."""

    model_config = ConfigDict(frozen=True)

    name: str  # names the node and the dir its sheets go to
    model: Path
    loras: dict[str, Path] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _plain_file_name(cls, name: str) -> str:
        return plain_file_name(name)


class PlannedPage(BaseModel):
    """What the plan node hands the draw nodes: the page's name and its crop plans. The name comes from the client
    and the draw nodes build file names from it, so it must be a plain file name."""

    page: str
    plans: list[CropPlan]

    @field_validator("page")
    @classmethod
    def _plain_file_name(cls, page: str) -> str:
        return plain_file_name(page)


def local_path(value: object) -> Path:
    """A file a node was given: a path, or a file dict holding a path or a ``/gradio_api/file=`` url.

    Raises:
        ValueError: If ``value`` names no local file.
    """
    if isinstance(value, str):
        return Path(value)
    if not isinstance(value, dict):
        raise ValueError(f"expected a file, got {type(value).__name__}")
    if value.get("path"):
        return Path(value["path"])
    url = str(value.get("url") or "")
    if not url.startswith(_FILE_URL):
        raise ValueError(f"expected a local file, got {url!r}")
    return Path(unquote(url.removeprefix(_FILE_URL)))


def media(path: Path) -> dict[str, object]:
    """``path`` as the file dict the canvas renders media from."""
    return {"path": str(path), "url": f"{_FILE_URL}{client_utils.encode_file_path(str(path))}", "is_file": True}


def _style(value: object) -> PageStyle:
    """A style as the canvas passes it (a dict) or the app view does (its JSON text)."""
    if isinstance(value, str):
        return PageStyle.model_validate_json(value)
    return PageStyle.model_validate(value)


def plan_node(template: object, text: str, style: object) -> dict[str, Any]:
    """The plan node: the template refilled with ``text`` (blank: its own running text) and planned as crops in
    ``style``."""
    path = local_path(template)
    page_text = text.strip() or running_text(load_xml_layout(path))
    planner = PagePlanBlocks().init_pipeline()
    plans = planner(template=path, text=page_text, style=_style(style), output="crop_plans")
    return PlannedPage(page=path.stem, plans=plans).model_dump(mode="json")


def crop_sheet(crops: list[DrawnCrop]) -> Image.Image:
    """The crops side by side at :data:`SHEET_HEIGHT`, each with its line boxes outlined."""
    panels: list[Image.Image] = []
    for crop in crops:
        panel = crop.image.convert("RGB").resize(
            (round(crop.image.width * SHEET_HEIGHT / crop.image.height), SHEET_HEIGHT)
        )
        draw = ImageDraw.Draw(panel)
        for line in crop.plan.lines:
            draw.rectangle(pixel_box(line.bbox, panel.size), outline=(220, 40, 40))
        panels.append(panel)
    sheet = Image.new("RGB", (sum(p.width for p in panels) + SHEET_GAP * (len(panels) - 1), SHEET_HEIGHT), "white")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width + SHEET_GAP
    return sheet


class DrawNode:
    """A drawer's draw node: every planned crop drawn by its crop pipeline, written to ``<out>/<drawer>/`` as a sheet
    of the crops and a zip of the crops with their labels.

    Drawers that share a model share its pipeline (``pipe``) with every LoRA they use loaded as a named adapter; a
    node activates its own adapters, and holds ``lock`` while it draws so one model runs on the device at a time.

    Args:
        drawer: The model and LoRAs this node draws with.
        pipe: Its model's crop pipeline (:func:`diffu_page.pipeline.crop_blocks`).
        settings: How every crop is drawn (the steps come from the node's input).
        out: Where the sheets and zips go.
        lock: Held while drawing, shared by every node of a studio.
    """

    def __init__(
        self,
        drawer: Drawer,
        pipe: ModularPipeline,
        *,
        settings: DrawSettings,
        out: Path,
        lock: threading.Lock,
    ) -> None:
        self._drawer = drawer
        self._pipe = pipe
        self._settings = settings
        self._out = out / drawer.name
        self._lock = lock

    def _activate_loras(self) -> None:
        if not self._pipe.get_list_adapters():  # the model has no LoRA loaded: nothing to switch
            return
        if self._drawer.loras:
            self._pipe.enable_lora()
            self._pipe.set_adapters(list(self._drawer.loras))
        else:
            self._pipe.disable_lora()

    def _draw(self, page: PlannedPage, seed: int, steps: int) -> list[DrawnCrop]:
        settings = self._settings.model_copy(update={"steps": steps})
        crops: list[DrawnCrop] = []
        with self._lock:
            self._activate_loras()
            for k, plan in enumerate(page.plans):
                name, crop_seed = crop_identity(page.page, seed, k)
                crops.append(draw_crop(self._pipe, plan, name=name, seed=crop_seed, reference=None, settings=settings))
        return crops

    def __call__(self, plans: object, seed: float, steps: float) -> tuple[dict[str, object], dict[str, object]]:
        page = PlannedPage.model_validate_json(plans) if isinstance(plans, str) else PlannedPage.model_validate(plans)
        crops = self._draw(page, int(seed), int(steps))
        self._out.mkdir(parents=True, exist_ok=True)
        stem = f"{page.page}_{int(seed)}"
        sheet = self._out / f"{stem}.jpg"
        crop_sheet(crops).save(sheet, quality=JPEG_QUALITY)
        archive = self._out / f"{stem}.zip"
        with zipfile.ZipFile(archive, "w") as files:
            for crop in crops:
                files.writestr(f"{crop.name}.jpg", jpeg_bytes(crop.image))
                files.writestr(f"{crop.name}.xml", crop.labels_xml)
        return media(sheet), media(archive)


def load_draw_nodes(drawers: list[Drawer], *, settings: DrawSettings, out: Path) -> dict[str, DrawNode]:
    """A draw node per drawer, by its bound function's name (:func:`diffu_studio.graph.draw_fn`); each model is loaded
    once, with every LoRA its drawers use.

    Raises:
        ValueError: If two drawers share a name (it names their node), two drawers name the same adapter of one model
            with different LoRA dirs, or LoRAs are to be fused into a model whose drawers switch them per drawer.
    """
    shared_names = sorted(name for name, count in Counter(drawer.name for drawer in drawers).items() if count > 1)
    if shared_names:
        raise ValueError(f"two drawers share each of the names {shared_names}: a name is its draw node's")
    loras_of: dict[Path, dict[str, Path]] = {}
    for drawer in drawers:
        model_loras = loras_of.setdefault(drawer.model, {})
        for adapter, lora_dir in drawer.loras.items():
            if model_loras.setdefault(adapter, lora_dir) != lora_dir:
                raise ValueError(
                    f"adapter {adapter!r} of {drawer.model} names two LoRAs: {model_loras[adapter]}, {lora_dir}"
                )
    switched = [model for model in loras_of if len({frozenset(d.loras) for d in drawers if d.model == model}) > 1]
    if settings.fuse_loras and switched:
        raise ValueError(f"the drawers of {switched} switch LoRAs per drawer, which fused LoRAs cannot: draw unfused")
    pipes = {
        model: load_pipeline(model, crop_blocks, loras=loras, settings=settings) for model, loras in loras_of.items()
    }
    lock = threading.Lock()
    return {
        draw_fn(drawer.name): DrawNode(drawer, pipes[drawer.model], settings=settings, out=out, lock=lock)
        for drawer in drawers
    }
