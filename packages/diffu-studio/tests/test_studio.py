"""The studio on CPU with diffusers' tiny FLUX.3 test model, a tiny diffu repo and the real HTR read-back: the canvas served
by gr.Workflow and called through gradio_client draws one plan of a page with every drawer, and drawers that share a
model switch its LoRAs."""

from __future__ import annotations

import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path

import gradio as gr
import pytest
import torch
from diffusers import Flux3Transformer2DModel
from gradio_client import Client, handle_file
from PIL import Image

from diffu_page.captions.look import ScanLook
from diffu_page.captions.prompt import LineRecord
from diffu_page.captions.provenance import Provenance
from diffu_page.layout import load_xml_layout, refill, running_text
from diffu_page.pipeline import DrawSettings
from diffu_page.plan import CropPlan, PageStyle, plan_page
from diffu_studio.nodes import SHEET_GAP, SHEET_HEIGHT, Drawer, PlannedPage, load_draw_nodes
from diffu_studio.studio import build_studio

TINY_SETTINGS = DrawSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
STYLE = PageStyle(
    provenance=Provenance(title="the archive", years="1700"), look=ScanLook(greyscale=False, paper="cream", ink="brown")
)
LORA_MODULE = "single_transformer_blocks.0.attn.to_q"


@pytest.fixture
def studio(tiny_flux3: Path, tiny_diffu: Path, tmp_path: Path) -> Iterator[gr.Workflow]:
    drawers = [Drawer(name="flux3", model=tiny_flux3), Drawer(name="diffu", model=tiny_diffu)]
    app = build_studio(drawers, settings=TINY_SETTINGS, out=tmp_path)
    app.launch(prevent_thread_lock=True, quiet=True, allowed_paths=[str(tmp_path)])
    yield app
    app.close()


def test_studio_draws_one_plan_of_a_page_with_every_drawer(studio: gr.Workflow, two_column_page: Path) -> None:
    client = Client(studio.local_url, verbose=False)

    flux3_sheet, flux3_zip, diffu_sheet, diffu_zip = client.predict(
        handle_file(two_column_page), "", STYLE.model_dump_json(), 3, 2, api_name="/flux3_crops"
    )

    layout = load_xml_layout(two_column_page)
    plans = plan_page(refill(layout, [running_text(layout)]), STYLE)
    names = [f"two_columns_{k}" for k in range(len(plans))]
    widths = [round(plan.width * SHEET_HEIGHT / plan.height) for plan in plans]
    for sheet, archive in ((flux3_sheet, flux3_zip), (diffu_sheet, diffu_zip)):
        assert Image.open(sheet).size == (sum(widths) + SHEET_GAP * (len(plans) - 1), SHEET_HEIGHT)
        with zipfile.ZipFile(archive) as files:
            assert files.namelist() == [f"{name}.{ext}" for name in names for ext in ("jpg", "xml")]
    assert Image.open(flux3_sheet).tobytes() != Image.open(diffu_sheet).tobytes()  # each drawer drew its own


def _small_page() -> dict[str, object]:
    plan = CropPlan(
        rect=(0, 0, 64, 32),
        width=64,
        height=32,
        caption="A handwritten page.",
        lines=[LineRecord(id="Sv_Text_1", text="Göteborgs poliskammare", role="", bbox=[0, 0, 1000, 1000])],
        reading_order=[0],
    )
    return PlannedPage(page="p", plans=[plan]).model_dump(mode="json")


def _pixels(sheet: dict[str, object]) -> bytes:
    with Image.open(str(sheet["path"])) as image:
        return image.tobytes()


def test_draw_nodes_on_one_model_switch_its_loras_per_drawer(
    tiny_flux3: Path, write_nemo_lora: Callable[..., Path], tmp_path: Path
) -> None:
    config = Flux3Transformer2DModel.load_config(tiny_flux3, subfolder="transformer")
    width = config["num_attention_heads"] * config["attention_head_dim"]
    torch.manual_seed(0)
    lora = write_nemo_lora(LORA_MODULE, torch.randn(2, width), torch.randn(width, 2), alpha=4)
    shared = load_draw_nodes(
        [Drawer(name="base", model=tiny_flux3), Drawer(name="v2", model=tiny_flux3, loras={"v2": lora})],
        settings=TINY_SETTINGS,
        out=tmp_path / "shared",
    )
    plain = load_draw_nodes([Drawer(name="plain", model=tiny_flux3)], settings=TINY_SETTINGS, out=tmp_path / "plain")

    with_lora = _pixels(shared["draw_v2"](_small_page(), 3, 2)[0])
    base = _pixels(shared["draw_base"](_small_page(), 3, 2)[0])

    assert base == _pixels(plain["draw_plain"](_small_page(), 3, 2)[0])
    assert with_lora != base
