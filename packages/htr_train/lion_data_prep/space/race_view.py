"""RaceView, a custom Gradio component: three readers' readings replayed at their measured pace, then Jev's scores
line by line. Its value is a `Race` payload (app.py); the drawing is in race_view.js and race_view.css."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

_HERE = Path(__file__).parent
HINT = (
    "Pick a region and press Read: the three readers run one after the other, and each reading is replayed here at "
    "the pace it was measured."
)
TEMPLATE = f'<div class="rv-root"><p class="rv-hint">{HINT}</p></div>'


class RaceView(gr.HTML):
    """Replays a `Race`: a bar and a text lane per reader, growing as the reader produced its tokens, and one row per
    line with Jev's scores. Readings are written as text, never as HTML."""

    def __init__(self, value: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(
            value=value,
            html_template=TEMPLATE,
            css_template=(_HERE / "race_view.css").read_text(encoding="utf-8"),
            js_on_load=(_HERE / "race_view.js").read_text(encoding="utf-8"),
            apply_default_css=False,
            **kwargs,
        )

    def api_info(self) -> dict[str, Any]:
        return {"type": "object", "title": "Race", "description": "Each reader's reading and timings, and Jev's scores"}
