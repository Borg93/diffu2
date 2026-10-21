"""The studio end to end on CPU: served by Gradio with the page pipeline on diffusers' tiny FLUX.3 test model and the real
HTR read-back, and called through gradio_client by the form's field names. A template given no text is drawn with its
own, and every crop comes back with a zip of the crops and their labels."""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path

import gradio as gr
import pytest
from gradio_client import Client, handle_file

from diffu_page.layout import load_xml_layout, refill, running_text
from diffu_page.pipeline import DrawSettings, load_pipeline, page_blocks
from diffu_page.plan import plan_page
from diffu_studio.studio import PageRequest, build_studio

TINY_SETTINGS = DrawSettings(steps=2, guidance_scale=4.0, text_layers=(1,), device="cpu")
STYLE_FIELDS = {"title": "the archive", "years": "1700", "paper": "cream", "ink": "brown", "greyscale": False}


@pytest.fixture
def studio(tiny_flux3: Path, tmp_path: Path) -> Iterator[gr.Blocks]:
    pipe = load_pipeline(tiny_flux3, page_blocks, loras={}, settings=TINY_SETTINGS)
    app = build_studio(pipe, settings=TINY_SETTINGS, out=tmp_path)
    app.launch(prevent_thread_lock=True, quiet=True)
    yield app
    app.close()


def test_studio_draws_a_template_with_its_own_text_and_zips_each_crop_with_its_labels(
    studio: gr.Blocks, two_column_page: Path
) -> None:
    client = Client(studio.local_url, verbose=False)

    gallery, archive = client.predict(
        template=handle_file(two_column_page), text=" ", seed=3, steps=2, **STYLE_FIELDS, api_name="/draw"
    )

    layout = load_xml_layout(two_column_page)
    style = PageRequest(template=two_column_page, text="", seed=3, steps=2, **STYLE_FIELDS).style()
    names = [f"two_columns_{k}" for k in range(len(plan_page(refill(layout, [running_text(layout)]), style)))]
    with zipfile.ZipFile(archive) as files:
        assert files.namelist() == [f"{name}.{ext}" for name in names for ext in ("jpg", "xml")]
    assert [item["caption"].split(":")[0] for item in gallery] == names
