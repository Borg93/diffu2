"""The studio on CPU with two tiny MOSS-Image models (conftest's tiny MOSS and its reference-trained export) and the
real HTR read-back: the canvas served by gr.Workflow and called through gradio_client draws one plan of a page with
every drawer, and drawers it could not tell apart or draw with are refused before any model loads."""

from __future__ import annotations

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pytest
from gradio_client import Client, handle_file
from PIL import Image
from pydantic import ValidationError

from diffu_page.captions.prompt import PageStyle
from diffu_page.layout import load_xml_layout, refill, running_text
from diffu_page.pipeline import DrawSettings
from diffu_page.plan import plan_page
from diffu_studio.graph import edge, studio_graph
from diffu_studio.nodes import SHEET_GAP, SHEET_HEIGHT, Drawer, PlannedPage, load_draw_nodes
from diffu_studio.studio import build_studio

TINY_SETTINGS = DrawSettings(steps=2, device="cpu")


@pytest.fixture
def studio(tiny_moss: Path, tiny_reference_moss: Path, tmp_path: Path) -> Iterator[gr.Workflow]:
    drawers = [Drawer(name="base", model=tiny_moss), Drawer(name="trained", model=tiny_reference_moss)]
    app = build_studio(drawers, settings=TINY_SETTINGS, out=tmp_path)
    app.launch(prevent_thread_lock=True, quiet=True, allowed_paths=[str(tmp_path)])
    yield app
    app.close()


def test_studio_draws_one_plan_of_a_page_with_every_drawer(
    studio: gr.Workflow, two_column_page: Path, page_style: PageStyle
) -> None:
    assert studio.local_url is not None
    client = Client(studio.local_url, verbose=False)

    base_sheet, base_zip, trained_sheet, trained_zip = client.predict(
        handle_file(two_column_page), "", page_style.model_dump_json(), 3, 2, api_name="/base_crops"
    )

    layout = load_xml_layout(two_column_page)
    plans = plan_page(refill(layout, running_text(layout)), page_style)
    names = [f"two_columns_{k}" for k in range(len(plans))]
    widths = [round(plan.width * SHEET_HEIGHT / plan.height) for plan in plans]
    for sheet, archive in ((base_sheet, base_zip), (trained_sheet, trained_zip)):
        assert Image.open(sheet).size == (sum(widths) + SHEET_GAP * (len(plans) - 1), SHEET_HEIGHT)
        with zipfile.ZipFile(archive) as files:
            assert files.namelist() == [f"{name}.{ext}" for name in names for ext in ("jpg", "xml")]
    assert Image.open(base_sheet).tobytes() != Image.open(trained_sheet).tobytes()  # each drawer drew its own


@pytest.mark.parametrize("page", [pytest.param("../../escaped", id="path"), pytest.param("..", id="parent")])
def test_planned_page_refuses_a_name_that_would_write_outside_the_out_dir(page: str) -> None:
    with pytest.raises(ValidationError, match="expected a plain file name"):
        PlannedPage.model_validate({"page": page, "plans": []})


def test_drawer_refuses_a_name_that_would_put_its_sheets_outside_the_out_dir() -> None:
    with pytest.raises(ValidationError, match="expected a plain file name"):
        Drawer(name="../base", model=Path("model"))


def test_edge_refuses_to_wire_ports_of_two_types(page_style: PageStyle) -> None:
    plan, draw = studio_graph(["base"], page_style).operators

    with pytest.raises(ValueError, match="op_plan.plans carries json, but op_draw_base.seed takes number"):
        edge(plan, "plans", draw, "seed")


def test_draw_nodes_refuse_two_drawers_of_one_name_before_loading(tmp_path: Path) -> None:
    drawers = [Drawer(name="v", model=Path("a")), Drawer(name="v", model=Path("b"))]

    with pytest.raises(ValueError, match=r"share each of the names \['v'\]"):
        load_draw_nodes(drawers, settings=TINY_SETTINGS, out=tmp_path)


def test_draw_nodes_refuse_a_model_directory_that_is_not_moss_images_before_loading(
    tiny_moss: Path, tiny_flux3: Path, tmp_path: Path
) -> None:
    drawers = [Drawer(name="moss", model=tiny_moss), Drawer(name="flux3", model=tiny_flux3)]

    with pytest.raises(ValueError, match=re.escape(f"flux3: {tiny_flux3} is no MOSS-Image model directory")):
        load_draw_nodes(drawers, settings=TINY_SETTINGS, out=tmp_path)
