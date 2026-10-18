"""The Guide tab — the interactive ``diffusion_intuition`` explainer, embedded in an isolated iframe."""

from __future__ import annotations

import base64
from pathlib import Path

import gradio as gr

_GUIDE_PATH = Path(__file__).parent.parent / "assets" / "guide.html"


def _guide_html() -> str:
    """Embed the guide in a ``data:`` URI iframe so its own <style>/<script> can't leak into the Gradio DOM."""
    try:
        raw = _GUIDE_PATH.read_bytes()
    except OSError:
        return "<p style='padding:24px;color:#5c6675'>Guide not bundled in this build.</p>"
    b64 = base64.b64encode(raw).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        'style="width:100%;height:82vh;border:1px solid #e6e1d6;border-radius:14px;background:#f7f6f2" '
        'title="diffu — the map and the machine"></iframe>'
    )


def build_guide_view() -> None:
    """Emit the guide iframe. No style row here — the Guide tab is just the explainer."""
    gr.HTML(_guide_html())
