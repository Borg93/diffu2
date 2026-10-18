"""The style-reference row: a drop target for the writer's hand + a gallery of real line crops to pick from.

No ``from __future__ import annotations`` — Gradio introspects the real annotation on ``pick`` to inject the
``gr.SelectData`` for the gallery click.
"""

import gradio as gr
from PIL import Image


def build_style_row(gallery_paths: list[str] | None) -> gr.Image:
    """Build the shared style-reference image + gallery picker; return the ``ref_state`` image both modes read."""
    with gr.Row(equal_height=True):  # equalises the image + gallery heights — no hardcoded height
        ref_state = gr.Image(
            type="pil",
            label="Style reference (writer's hand) — drop a line crop or pick one from the gallery →",
            height=200,  # matches the gallery strip
            scale=2,
        )
        gallery = (
            gr.Gallery(
                gallery_paths,
                columns=1,  # one full-width strip per line crop (cell aspect forced wide via _CSS in ui.py)
                height=200,  # ~3 strips visible, rest scrolls; equal_height matches the image column to it
                object_fit="contain",
                elem_id="style_gallery",
                allow_preview=False,
                label="pick a reference line",
                scale=3,
            )
            if gallery_paths
            else None
        )
    if gallery is not None and gallery_paths:

        def pick(evt: gr.SelectData) -> Image.Image:
            return Image.open(gallery_paths[evt.index]).convert("RGB")

        gallery.select(pick, None, ref_state)
    return ref_state
