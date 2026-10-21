"""Layout stage — carve a page into regions and flow text into line boxes.

A :class:`LineBox` is the entire interface to the renderer/compositor: coordinates + the text for
that line. Boxes come from a parameterized template here, or from real ALTO/PAGE-XML
(:func:`load_xml_layout`). Every page format reduces to: carve the page into rectangular regions,
flow lines down each, optionally record furniture rules + page elements (page numbers, marginalia,
headers). No imaging here; this is pure geometry.

:class:`PageSpec` is the **generation control surface** — every knob you parameterize. Randomize its
fields per page (see ``__main__``) for docTR-style variety; the engine is deterministic given
``(spec, seed)``.

Split into cohesive submodules — ``model`` (the value objects), ``flow`` (wrap + flow geometry),
``templates`` (every page format + :func:`build_layout`), ``edits`` (manuscript-edit passes) and
``xml_replay`` (:func:`load_xml_layout`) — this package re-exports the old flat-module surface.
"""

from __future__ import annotations

from .edits import (
    _BODY_REGIONS as _BODY_REGIONS,
)
from .edits import (
    _INITIAL_REGIONS as _INITIAL_REGIONS,
)
from .edits import (
    _INSERTS as _INSERTS,
)
from .edits import (
    _apply_drop_cap as _apply_drop_cap,
)
from .edits import (
    _apply_edits as _apply_edits,
)
from .edits import (
    is_body,
)
from .flow import (
    _box as _box,
)
from .flow import (
    _chars_per_line as _chars_per_line,
)
from .flow import (
    _fit_to_width as _fit_to_width,
)
from .flow import (
    _flow as _flow,
)
from .flow import (
    _flow_paragraphs as _flow_paragraphs,
)
from .flow import (
    _margin_key_box as _margin_key_box,
)
from .flow import (
    _normalize_line_heights as _normalize_line_heights,
)
from .flow import (
    _page_elements as _page_elements,
)
from .flow import (
    _wrap as _wrap,
)
from .model import (
    Align,
    Column,
    Layout,
    LineBox,
    Measure,
    PageSpec,
    Rule,
    TableColumn,
    validate_layout,
)
from .templates import (
    _DEFAULT_TABLE_COLUMNS as _DEFAULT_TABLE_COLUMNS,
)
from .templates import (
    _LANDSKAP_DUES as _LANDSKAP_DUES,
)
from .templates import (
    _LANDSKAP_KINDS as _LANDSKAP_KINDS,
)
from .templates import (
    _LANDSKAP_PLACES as _LANDSKAP_PLACES,
)
from .templates import (
    _MONTHS as _MONTHS,
)
from .templates import (
    _TEMPLATES as _TEMPLATES,
)
from .templates import (
    TEMPLATES,
    LeafPrefer,
    build_layout,
    to_single_leaf,
)
from .templates import (
    _blank_leaf as _blank_leaf,
)
from .templates import (
    _columns as _columns,
)
from .templates import (
    _key_value as _key_value,
)
from .templates import (
    _landskap as _landskap,
)
from .templates import (
    _ledger as _ledger,
)
from .templates import (
    _letter as _letter,
)
from .templates import (
    _multi_column as _multi_column,
)
from .templates import (
    _register as _register,
)
from .templates import (
    _semi_table as _semi_table,
)
from .templates import (
    _single_column as _single_column,
)
from .templates import (
    _spread as _spread,
)
from .templates import (
    _spread_leaf as _spread_leaf,
)
from .text_refill import refill, running_text
from .xml_replay import (
    _PAGE_TYPE_REGION as _PAGE_TYPE_REGION,
)
from .xml_replay import (
    _line_region as _line_region,
)
from .xml_replay import (
    _local_tag as _local_tag,
)
from .xml_replay import (
    _separator_rules as _separator_rules,
)
from .xml_replay import (
    load_xml_layout,
)

__all__ = [
    "Align",
    "Column",
    "Layout",
    "LeafPrefer",
    "LineBox",
    "Measure",
    "PageSpec",
    "Rule",
    "TEMPLATES",
    "TableColumn",
    "build_layout",
    "is_body",
    "load_xml_layout",
    "refill",
    "running_text",
    "to_single_leaf",
    "validate_layout",
]
