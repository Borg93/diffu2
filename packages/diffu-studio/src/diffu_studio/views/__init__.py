"""Studio view builders — one module per screen region, each owns its widgets + event wiring."""

from diffu_studio.views.guide_view import build_guide_view
from diffu_studio.views.line_view import build_line_view
from diffu_studio.views.page_view import build_page_view
from diffu_studio.views.style_view import build_style_row

__all__ = ["build_guide_view", "build_line_view", "build_page_view", "build_style_row"]
