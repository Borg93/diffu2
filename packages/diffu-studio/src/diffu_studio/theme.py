"""The Diffu Studio Gradio theme — the warm archival look of the ``diffusion_intuition`` guide.

Gradio 6 dropped the ``theme=`` kwarg from ``gr.Blocks``; it now lives on ``.launch()`` (verified against the
installed 6.18). So the launch sites (``__main__`` locally, ``app.py`` on the Space) pass ``diffu_theme()``.

Palette is lifted straight from the guide's ``:root`` tokens: teal ``#0e7c66`` primary, amber accent, warm
paper ``#f7f6f2`` canvas, Space Grotesk / JetBrains Mono. The ``*_dark`` variants are pinned to the SAME light
values so the paper aesthetic holds whether the browser requests light or dark — the guide is light-only.
"""

from __future__ import annotations

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

# guide :root tokens
_TEAL = "#0e7c66"
_TEAL_HOVER = "#0c6c58"
_PAPER = "#f7f6f2"
_PANEL = "#ffffff"
_PANEL2 = "#fbfaf7"
_INPUT = "#fdfcf9"
_BORDER = "#e6e1d6"
_BORDER2 = "#cfc8ba"
_INK = "#1f2733"
_MUTED = "#5c6675"


def diffu_theme() -> gr.Theme:
    """A ``gr.themes.Base`` tuned to the guide's palette + fonts. Pass to ``demo.launch(theme=...)``."""
    return gr.themes.Base(
        primary_hue=colors.emerald,  # teal #0e7c66 ≈ emerald 700 — drives buttons, sliders, the loader
        secondary_hue=colors.amber,
        neutral_hue=colors.stone,  # warm gray to match the #e6e1d6 borders / #f7f6f2 paper
        radius_size=sizes.radius_lg,
        text_size=sizes.text_md,
        spacing_size=sizes.spacing_md,
        font=(fonts.GoogleFont("Space Grotesk"), "system-ui", "-apple-system", "sans-serif"),
        font_mono=(fonts.GoogleFont("JetBrains Mono"), "ui-monospace", "Menlo", "monospace"),
    ).set(
        # canvas + text
        body_background_fill=_PAPER,
        body_background_fill_dark=_PAPER,
        body_text_color=_INK,
        body_text_color_dark=_INK,
        body_text_color_subdued=_MUTED,
        body_text_color_subdued_dark=_MUTED,
        # panels / blocks
        background_fill_primary=_PANEL,
        background_fill_primary_dark=_PANEL,
        background_fill_secondary=_PANEL2,
        background_fill_secondary_dark=_PANEL2,
        block_background_fill=_PANEL,
        block_background_fill_dark=_PANEL,
        block_border_color=_BORDER,
        block_border_color_dark=_BORDER,
        block_border_width="1px",
        block_label_background_fill=_PANEL2,
        block_label_background_fill_dark=_PANEL2,
        block_label_text_color=_MUTED,
        block_label_text_color_dark=_MUTED,
        block_title_text_color=_INK,
        block_title_text_color_dark=_INK,
        block_shadow="0 1px 3px rgba(31,39,51,.06)",
        border_color_primary=_BORDER,
        border_color_primary_dark=_BORDER,
        border_color_accent=_TEAL,
        color_accent_soft="rgba(14,124,102,.08)",
        # inputs
        input_background_fill=_INPUT,
        input_background_fill_dark=_INPUT,
        input_border_color=_BORDER,
        input_border_color_dark=_BORDER,
        input_border_color_focus=_TEAL,
        input_border_color_focus_dark=_TEAL,
        # primary (teal) + secondary (warm paper) buttons
        button_primary_background_fill=_TEAL,
        button_primary_background_fill_dark=_TEAL,
        button_primary_background_fill_hover=_TEAL_HOVER,
        button_primary_background_fill_hover_dark=_TEAL_HOVER,
        button_primary_border_color=_TEAL,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill=_PANEL,
        button_secondary_background_fill_dark=_PANEL,
        button_secondary_background_fill_hover=_PANEL2,
        button_secondary_border_color=_BORDER2,
        button_secondary_border_color_dark=_BORDER2,
        button_secondary_text_color=_INK,
        button_secondary_text_color_dark=_INK,
        # sliders / loader pick up primary_hue automatically; pin the slider track to the exact teal
        slider_color=_TEAL,
        slider_color_dark=_TEAL,
    )
