"""The Gradio Blocks control panel — a top-level **Studio** tab (whole-page ⟷ single-line toggle) + **Guide**.

This module is deliberately slim: it only assembles the top-level layout. The widgets + event wiring live in
:mod:`diffu_studio.views`, the request-boundary forms in :mod:`diffu_studio.forms` /
:mod:`diffu_studio.controls`, and the engine-bound services in :mod:`diffu_studio.handlers`.
"""

import gradio as gr

from diffu_studio.engine import StudioEngine
from diffu_studio.views import build_guide_view, build_line_view, build_page_view, build_style_row

# Two paragraphs (blank line between) so the default page DEMONSTRATES the paragraph gap: each block flows
# and re-wraps to the column, and a paragraph break separates them — matching the Text box's tooltip.
_SAMPLE_TEXT = (
    "Skattebref af den 23 Mars 1854 för bönderne Petter Hirvonen och Matts "
    "Hulkko å hälften af Sju Fyratioåttondedels mantal i Jjunga by.\n"
    "\n"
    "Beslöts att ärendet skulle företagas till pröfning vid nästa "
    "sammanträde, och att parterna dertill behörigen kallas."
)

# Line crops are wide (≈8–12:1), so give every gallery cell a full-width wide aspect and letterbox the
# image inside it — otherwise Gradio's square default crops each line to an unreadable sliver. Injected
# via gr.HTML (works on every Gradio version; Gradio 6 DROPS the Blocks `css=` kwarg — it must not be used).
_CSS = (
    "<style>"
    "#style_gallery .grid-container { grid-template-columns: 1fr !important; }"
    "#style_gallery .thumbnail-item { aspect-ratio: 8 / 1 !important; width: 100% !important; }"
    "#style_gallery .thumbnail-item img { object-fit: contain !important; width: 100%; height: 100%; }"
    "</style>"
)


def build_blocks(
    engine: StudioEngine, *, gallery_paths: list[str] | None, default_ckpt: str | None
) -> gr.Blocks:
    """Assemble the control panel bound to ``engine``. One fixed checkpoint — ``default_ckpt`` (no model picker)."""
    if default_ckpt is None:
        raise ValueError(
            "Diffu Studio needs a checkpoint, but none was found (pass --ckpt or fill checkpoints/)."
        )

    with gr.Blocks(title="Diffu Studio") as demo:
        gr.HTML(_CSS)  # style overrides — see _CSS; gr.HTML because Gradio 6 ignores Blocks(css=…)
        gr.Markdown("## ✒️ Diffu Studio")
        with gr.Tabs():
            with gr.Tab("✒️  Studio"):
                # One input card: what to imitate (style + gallery), which mode, and the text — all together.
                with gr.Group():
                    ref_state = build_style_row(gallery_paths)
                    with gr.Row(equal_height=True):
                        mode = gr.Radio(
                            ["Whole page", "Single line"],
                            value="Whole page",
                            label="Mode",
                            scale=1,
                            info="Whole page = a document; Single line = one line + read-back.",
                        )
                        text = gr.Textbox(
                            label="Text",
                            value=_SAMPLE_TEXT,
                            lines=3,
                            scale=3,
                            info="Whole page: blank line = new paragraph; single line-breaks flow together "
                            "and re-wrap. Single line: the text to render.",
                        )
                with gr.Group(visible=True) as page_group:
                    build_page_view(engine=engine, default_ckpt=default_ckpt, ref_state=ref_state, text=text)
                with gr.Group(visible=False) as single_group:
                    build_line_view(engine=engine, default_ckpt=default_ckpt, ref_state=ref_state, text=text)

                def toggle(chosen: str) -> tuple[dict, dict]:
                    return (
                        gr.update(visible=chosen == "Whole page"),
                        gr.update(visible=chosen == "Single line"),
                    )

                mode.change(toggle, mode, [page_group, single_group])
            with gr.Tab("📖  Guide"):
                build_guide_view()
    return demo
