"""Control-bags: map a group of live Gradio widgets to a validated form at the request boundary.

These are deliberately *plain classes*, not Pydantic models: they hold references to live ``gr`` components,
and Gradio's dict-inputs mode keys the event payload by the component OBJECT itself. A Pydantic model could
copy/re-wrap the value during validation and break that identity, so we store the components verbatim and only
build a Pydantic form (the structured data) inside :meth:`read`.

Each bag exposes ``components()`` (the input *set* that turns on Gradio dict-mode) and ``read(data)`` (the one
place that maps ``{component: value}`` back to a typed form). This is the single source of truth that replaces
the four hand-synced 24-wide positional lists the old handlers duplicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from diffu_studio.configs import SampleSettings
from diffu_studio.forms import AugmentForm, LineForm, PageForm

if TYPE_CHECKING:
    import gradio as gr

    type Component = gr.components.Component
    # Gradio's dict-inputs payload is dynamically typed ({component: raw widget value}); Pydantic validates it
    # at the boundary inside read(), so Any here is honest rather than a cast-fest.
    type Payload = dict[Component, Any]

# Presence gates → rendered as CHECKBOXES (deterministic on/off, no fake probability slider). They fire the
# ``.change`` event, not ``.release``.
_GATE_KNOBS = frozenset({
    "bed_prob", "microfilm_prob", "correction_prob", "stamp_prob", "fold_prob", "dust_prob",
    "clasp_prob", "glove_prob", "hole_prob", "curl_prob", "paraph_prob", "margin_mark_prob",
})  # fmt: skip
_PICKER_KNOBS = frozenset({"ink_color", "foxing_tint"})  # dropdown / colour — also fire .change
_CHANGE_KNOBS = (
    _GATE_KNOBS | _PICKER_KNOBS
)  # everything wired to .change; the rest are strength sliders (.release)
_PAGE_GATES = frozenset({"furniture_prob", "binding_prob"})  # page-level presence gates (also checkboxes)


def _coerce(value: Any) -> Any:
    """A checkbox bool → a float gate (True→1.0, False→0.0); everything else passes through unchanged."""
    return float(value) if isinstance(value, bool) else value


class AugmentControls:
    """The 24 scan-look widgets, keyed by their :class:`AugmentForm` field name.

    The component-reference accessors (``components``/``sliders``/``toggles``) are typed ``Any``: they feed
    Gradio's dynamically-typed event API (``.release``/``.change``/``inputs=``), which ty can't model
    per-component. ``read`` — the boundary that actually matters — returns a strongly-typed, validated form.
    """

    def __init__(self, by_name: dict[str, Component]) -> None:
        self._by_name = by_name

    def components(self) -> set[Any]:
        return set(self._by_name.values())

    def sliders(self) -> list[Any]:
        """Strength sliders — wired to ``.release`` (fire once the drag ends)."""
        return [c for name, c in self._by_name.items() if name not in _CHANGE_KNOBS]

    def toggles(self) -> list[Any]:
        """Presence checkboxes + the colour/ink pickers — wired to ``.change``."""
        return [c for name, c in self._by_name.items() if name in _CHANGE_KNOBS]

    def read(self, data: Payload) -> AugmentForm:
        return AugmentForm.model_validate({name: _coerce(data[c]) for name, c in self._by_name.items()})


class PageControls:
    """Every whole-page widget: the flat page knobs plus the nested :class:`AugmentControls`."""

    def __init__(self, by_name: dict[str, Component], augment: AugmentControls) -> None:
        self._by_name = by_name  # keys match PageForm fields (except ``augment``)
        self.augment = augment

    def components(self) -> set[Any]:
        return set(self._by_name.values()) | self.augment.components()

    def read(self, data: Payload) -> PageForm:
        # Only the page-level presence gates (furniture/binding checkboxes) coerce bool→float; typeset_furniture
        # stays a real bool.
        flat = {
            name: (_coerce(data[c]) if name in _PAGE_GATES else data[c]) for name, c in self._by_name.items()
        }
        return PageForm.model_validate({**flat, "augment": self.augment.read(data).model_dump()})


class LineControls:
    """The single-line widgets — text plus the sampling knobs (no model choice)."""

    def __init__(self, by_name: dict[str, Component]) -> None:
        self._by_name = by_name  # text, steps, cfg, auto, width

    def components(self) -> set[Any]:
        return set(self._by_name.values())

    def read(self, data: Payload) -> LineForm:
        return LineForm(
            text=data[self._by_name["text"]],
            settings=SampleSettings(
                steps=int(data[self._by_name["steps"]]),
                cfg_scale=data[self._by_name["cfg"]],
                auto_width=data[self._by_name["auto"]],
                width=int(data[self._by_name["width"]]),
            ),
        )
