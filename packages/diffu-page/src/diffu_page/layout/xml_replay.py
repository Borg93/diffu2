"""XML replay — build a layout from real ALTO or PAGE XML instead of a template.

Reads TextLine geometry + transcription, region semantics (TextRegion@type, or the ``structure {type:...}`` that
Transkribus writes into ``@custom`` / TableRegion) and separator furniture, namespace-agnostic across PAGE (PRImA)
and ALTO. The renderer/compositor consume the result identically to a template-built layout — that's the LineBox
seam.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from .flow import _normalize_line_heights
from .model import Layout, LineBox, PageSpec, Rule
from .templates import LeafPrefer, to_single_leaf

# PAGE ``TextRegion@type`` -> our region vocabulary (the reverse of labels._PAGE_REGION_TYPE); an absent or
# unrecognized type falls back to body (reads in normal flow) rather than the old opaque "xml".
_PAGE_TYPE_REGION: dict[str, str] = {
    "paragraph": "body",
    "heading": "header",
    "header": "header",
    "marginalia": "marginalia",
    "page-number": "page_number",
    "catch-word": "catchword",
    "signature-mark": "signature",
    "caption": "body",
    "footnote": "body",
    "drop-capital": "header",
    "other": "body",
}
_CUSTOM_STRUCTURE = re.compile(r"structure \{type:([^;}]+)")


def _local_tag(tag: str) -> str:
    """Local name of a possibly namespaced XML tag (``{ns}TextLine`` -> ``TextLine``)."""
    return tag.rsplit("}", 1)[-1]


def _line_region(line: ET.Element, parents: dict[ET.Element, ET.Element]) -> str:
    """Semantic region name for a ``TextLine`` from its nearest enclosing region ancestor.

    A cell in a PAGE ``TableRegion``/``TableCell`` reads as ``table``; a ``TextRegion`` maps by its type
    (paragraph->body, heading->header, marginalia, ...); anything else (e.g. an ALTO ``TextBlock``) is body.
    """
    cur = parents.get(line)
    while cur is not None:
        tag = _local_tag(cur.tag)
        if tag in ("TableRegion", "TableCell"):
            return "table"
        if tag == "TextRegion":
            return _PAGE_TYPE_REGION.get(_region_type(cur), "body")
        cur = parents.get(cur)
    return "body"


def _region_type(region: ET.Element) -> str:
    """A ``TextRegion``'s PAGE type: its ``@type``, else the ``structure {type:...}`` in its Transkribus ``@custom``."""
    structure = _CUSTOM_STRUCTURE.search(region.get("custom") or "")
    return (region.get("type") or (structure.group(1) if structure else "")).lower()


def _separator_rules(root: ET.Element) -> list[Rule]:
    """PAGE ``SeparatorRegion`` polygons -> ruled lines (real column/table separators), so a replayed layout
    draws its furniture. Orientation comes from the polygon bbox — wider than tall -> horizontal, else
    vertical — collapsed to the mid-line."""
    rules: list[Rule] = []
    for el in root.iter():
        if _local_tag(el.tag) != "SeparatorRegion":
            continue
        coords = next((c for c in el if _local_tag(c.tag) == "Coords"), None)
        points = coords.get("points") if coords is not None else None
        if not points:
            continue
        xy = [(int(p.split(",")[0]), int(p.split(",")[1])) for p in points.split() if "," in p]
        if len(xy) < 2:
            continue
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        if (x1 - x0) >= (y1 - y0):  # horizontal rule
            ym = (y0 + y1) // 2
            rules.append(Rule(x1=x0, y1=ym, x2=x1, y2=ym, width=2))
        else:  # vertical rule
            xm = (x0 + x1) // 2
            rules.append(Rule(x1=xm, y1=y0, x2=xm, y2=y1, width=2))
    return rules


def _page_dims(root: ET.Element) -> tuple[int, int] | None:
    """The real canvas size from the ``Page`` element (PAGE or ALTO attributes), if it carries one."""
    page = next((e for e in root.iter() if _local_tag(e.tag) == "Page"), None)
    if page is None:
        return None
    w = page.get("WIDTH") or page.get("imageWidth")
    h = page.get("HEIGHT") or page.get("imageHeight")
    if not (w and h):
        return None
    return int(w), int(h)


def _line_text_and_coords(el: ET.Element) -> tuple[str, list[tuple[int, int]]]:
    """One ``TextLine``'s transcription + polygon points, across the PAGE and ALTO encodings."""
    text = ""
    coords_pts: list[tuple[int, int]] = []
    for child in el.iter():
        t = _local_tag(child.tag)
        points = child.get("points")
        content = child.get("CONTENT")
        if t == "Unicode" and child.text:
            text = child.text.strip()
            continue
        if t == "Coords" and points:  # PAGE
            coords_pts = [(int(xy[0]), int(xy[1])) for xy in (p.split(",") for p in points.split())]
            continue
        if t == "String" and content:  # ALTO line text on the String element
            text = f"{text} {content}".strip()
    # ALTO geometry lives as HPOS/VPOS/WIDTH/HEIGHT attributes on the TextLine itself
    hpos = el.get("HPOS")
    if not coords_pts and hpos is not None:
        x, y = int(hpos), int(el.get("VPOS") or 0)
        w, h = int(el.get("WIDTH") or 0), int(el.get("HEIGHT") or 0)
        coords_pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    return text, coords_pts


def _line_from_element(
    el: ET.Element, parents: dict[ET.Element, ET.Element], spec: PageSpec
) -> LineBox | None:
    """A replayed ``LineBox`` for one ``TextLine`` element, or ``None`` when it lacks text or geometry."""
    text, coords_pts = _line_text_and_coords(el)
    if not coords_pts or not text:
        return None
    xs = [p[0] for p in coords_pts]
    ys = [p[1] for p in coords_pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return LineBox(
        text=text,
        x=x0,
        y=y0,
        w=x1 - x0,
        h=y1 - y0,
        baseline_y=int(y0 + spec.baseline_frac * (y1 - y0)),
        region=_line_region(el, parents),
    )


def load_xml_layout(
    xml_path: str, spec: PageSpec | None = None, *, single_page: bool = True, prefer: LeafPrefer = "denser"
) -> Layout:
    """Build a layout from a real ALTO or PAGE XML file (replay real layouts).

    Reads one ``TextLine`` per box: its bbox (or polygon) + transcription, its **region semantics** (from the
    enclosing ``TextRegion@type`` / ``TableRegion``, so a replayed marginal note or table cell reads and
    labels correctly instead of flattening to a single opaque region), and the page's **separator furniture**
    (PAGE ``SeparatorRegion`` -> ruled lines). Namespace-agnostic across PAGE (PRImA) and ALTO. The
    renderer/compositor consume this identically to a template-built layout — that's the LineBox seam.

    ``single_page`` (default) reduces a two-page-spread XML to one leaf via :func:`to_single_leaf` — we
    only train on single pages — choosing the leaf with ``prefer`` ('denser' | 'blank' | 'left' | 'right').
    """
    root = ET.parse(xml_path).getroot()
    parents = {child: parent for parent in root.iter() for child in parent}

    spec = spec or PageSpec()
    dims = _page_dims(root)
    if dims is not None:  # use the real canvas size so boxes land on-page
        spec = spec.model_copy(update={"width": dims[0], "height": dims[1]})

    boxes: list[LineBox] = []
    for el in root.iter():
        if _local_tag(el.tag) != "TextLine":
            continue
        line = _line_from_element(el, parents, spec)
        if line is not None:
            boxes.append(line)
    layout = Layout(spec=spec, boxes=_normalize_line_heights(boxes, spec), rules=_separator_rules(root))
    return to_single_leaf(layout, prefer) if single_page else layout
