"""Label data + pluggable export formats.

The compositor produces a cohesive :class:`PageLabels` value (lines, words, polygons,
transcriptions, regions, reading order). Output *formats* are decoupled from that data: each is a
plain ``PageLabels -> str`` function registered in :data:`EXPORTERS`. ALTO v4 is just one entry, not
a privileged format — add a format by writing a function and adding one dict line.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field

try:
    _VERSION = version("diffu-page")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    _VERSION = "0"

type Point = tuple[int, int]


class WordLabel(BaseModel):
    """One word's transcription + polygon (page coordinates)."""

    text: str
    polygon: list[Point]


class LineLabel(BaseModel):
    """One text line: transcription, polygon, baseline polyline, region, reading-order index, words."""

    text: str
    polygon: list[Point]
    baseline: list[Point] = Field(default_factory=list)  # left->right baseline points (empty if unknown)
    region: str = "body"
    order: int = 0
    words: list[WordLabel] = Field(default_factory=list)
    tag: str = ""  # editorial tag: "" | add (interlinear insertion) | del (struck/deleted) | unclear
    printed: bool = False  # typeset form-furniture (header/row number) vs handwriting -> PAGE production type


class PageLabels(BaseModel):
    """A page's full ground truth — the single value every exporter serializes."""

    width: int
    height: int
    image: str = ""
    lines: list[LineLabel] = Field(default_factory=list)


def _bbox(poly: list[Point]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _esc(s: str) -> str:
    return escape(s, {'"': "&quot;"})  # values land in XML attributes, so quotes must be entities too


# PAGE TextRegion @type uses a controlled vocabulary; map our region names onto it (default paragraph).
_PAGE_REGION_TYPE: dict[str, str] = {
    "body": "paragraph",
    "header": "heading",
    "marginalia": "marginalia",
    "margin_key": "marginalia",  # a keyed marginal note (reads interleaved with the body)
    "page_number": "page-number",
    "table_header": "heading",
    "table": "other",
    "row_number": "other",
    "kv_label": "paragraph",
    "kv_value": "paragraph",
    "xml": "paragraph",
    "catchword": "catch-word",
    "signature": "signature-mark",
    "reg_header": "heading",
    "rubric": "heading",
    "initial": "paragraph",  # an enlarged opening capital (versal) — part of the running text
}


def _page_type(region: str) -> str:
    if region.startswith("col"):  # col0/col1/... multi-column body
        return "paragraph"
    return _PAGE_REGION_TYPE.get(region, "paragraph")


def _ordered_regions(labels: PageLabels) -> list[tuple[str, list[LineLabel]]]:
    """Group lines by region, sort each region's lines by reading-order index, and return the regions
    themselves in reading order (by their first line's order). One source of truth for every exporter."""
    groups: dict[str, list[LineLabel]] = {}
    for line in labels.lines:
        groups.setdefault(line.region, []).append(line)
    for lines in groups.values():
        lines.sort(key=lambda ln: ln.order)
    return sorted(groups.items(), key=lambda kv: min(ln.order for ln in kv[1]))


def to_json(labels: PageLabels) -> str:
    """Universal JSON: page dims + a flat line list (polygon, text, region, order, words)."""
    return json.dumps(labels.model_dump(), ensure_ascii=False, indent=2)


_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_PAGE_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_PAGE_SCHEMA = f"{_PAGE_NS} {_PAGE_NS}/pagecontent.xsd"
# Synthetic ground truth has no real authoring moment; a fixed UTC placeholder keeps the schema-required
# Metadata timestamps valid AND the export byte-reproducible for a given page.
_CREATED = "1970-01-01T00:00:00Z"

# tag -> the PAGE TextStyle boolean that marks it (the schema-sanctioned way, alongside the Transkribus custom)
_TAG_STYLE = {
    "del": "strikethrough",
    "add": "superscript",
    "unclear": "underlined",
    "underline": "underlined",
}


def _pts(points: list[Point]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _page_words(region: str, line: LineLabel) -> str:
    """``<Word>`` elements (Coords + transcription) for a line's word boxes — empty if it has none."""
    return "\n".join(
        f'        <Word id="{_esc(region)}_l{line.order}_w{wi}">\n'
        f'          <Coords points="{_pts(wd.polygon)}"/>\n'
        f"          <TextEquiv><Unicode>{_esc(wd.text)}</Unicode></TextEquiv>\n"
        f"        </Word>"
        for wi, wd in enumerate(line.words)
    )


def to_page_xml(labels: PageLabels) -> str:
    """PRImA PAGE-XML (validates against the 2019 / 2013 content model): required Metadata timestamps, an
    explicit ``ReadingOrder``, one typed ``TextRegion`` (Coords) per region, and a ``TextLine`` (Coords,
    Baseline, ``Word`` boxes, TextEquiv) per line — with ``production`` marking handwriting vs print and a
    ``TextStyle`` flag (strikethrough/superscript) for struck/inserted edits."""
    regions = _ordered_regions(labels)
    refs = "\n".join(
        f'        <RegionRefIndexed index="{i}" regionRef="{_esc(region)}"/>'
        for i, (region, _) in enumerate(regions)
    )
    reading_order = (
        "    <ReadingOrder>\n"
        f'      <OrderedGroup id="ro">\n{refs}\n      </OrderedGroup>\n'
        "    </ReadingOrder>\n"
    )
    blocks: list[str] = []
    for region, lines in regions:
        rx, ry, rw, rh = _bbox([p for ln in lines for p in ln.polygon])
        rcoords = _pts([(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)])
        tl: list[str] = []
        for ln in lines:
            production = "printed" if ln.printed else "handwritten-cursive"
            base = f'        <Baseline points="{_pts(ln.baseline)}"/>\n' if ln.baseline else ""
            words = _page_words(region, ln)
            words = f"{words}\n" if words else ""
            custom = f' custom="structure {{type:{_esc(ln.tag)};}}"' if ln.tag else ""  # Transkribus-style
            style = f'        <TextStyle {_TAG_STYLE[ln.tag]}="true"/>\n' if ln.tag in _TAG_STYLE else ""
            tl.append(
                f'      <TextLine id="{_esc(region)}_l{ln.order}" production="{production}"{custom}>\n'
                f'        <Coords points="{_pts(ln.polygon)}"/>\n'
                f"{base}{words}"
                f"        <TextEquiv><Unicode>{_esc(ln.text)}</Unicode></TextEquiv>\n"
                f"{style}"
                f"      </TextLine>"
            )
        blocks.append(
            f'    <TextRegion id="{_esc(region)}" type="{_page_type(region)}">\n'
            f'      <Coords points="{rcoords}"/>\n' + "\n".join(tl) + "\n    </TextRegion>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<PcGts xmlns="{_PAGE_NS}" xmlns:xsi="{_PAGE_XSI}" xsi:schemaLocation="{_PAGE_SCHEMA}">\n'
        "  <Metadata>\n"
        "    <Creator>diffu_page</Creator>\n"
        f"    <Created>{_CREATED}</Created>\n"
        f"    <LastChange>{_CREATED}</LastChange>\n"
        "  </Metadata>\n"
        f'  <Page imageFilename="{_esc(labels.image)}" imageWidth="{labels.width}" '
        f'imageHeight="{labels.height}">\n' + reading_order + "\n".join(blocks) + "\n  </Page>\n</PcGts>\n"
    )


_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v4#"
_ALTO_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_ALTO_SCHEMA = "http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-4.xsd"


def _alto_description(image: str) -> str:
    """The ALTO ``<Description>`` provenance block in the HTRFlow shape, but honest about what this is —
    synthetic ground truth from diffu_page, not an HTR recognition result. ``processingDateTime`` is left
    out on purpose so the export stays byte-reproducible for a given page (ALTO makes it optional)."""
    return (
        "  <Description>\n"
        "    <MeasurementUnit>pixel</MeasurementUnit>\n"
        f"    <sourceImageInformation><fileName>{image}</fileName></sourceImageInformation>\n"
        '    <Processing ID="synthesis">\n'
        "      <processingStepDescription>Synthesis</processingStepDescription>\n"
        "      <processingStepSettings>renderer: diffu_page; labels: ground-truth</processingStepSettings>\n"
        "      <processingSoftware>\n"
        "        <softwareCreator>diffu_page</softwareCreator>\n"
        "        <softwareName>diffu_page</softwareName>\n"
        f"        <softwareVersion>{_VERSION}</softwareVersion>\n"
        "        <applicationDescription>synthetic handwriting pages with ground-truth labels"
        "</applicationDescription>\n"
        "      </processingSoftware>\n"
        "    </Processing>\n"
        "  </Description>\n"
    )


def _alto_reading_order(regions: list[tuple[str, list[LineLabel]]]) -> str:
    """ALTO ``<ReadingOrder>``: a nested ``OrderedGroup`` per region, an ``ElementRef`` per line (REF =
    the line's ``TextLine`` ID) — the htrflow shape, so a reader can recover line sequence without geometry."""
    groups: list[str] = []
    for region, lines in regions:
        refs = "\n".join(
            f'        <ElementRef ID="ro_{_esc(region)}_l{ln.order}" REF="{_esc(region)}_l{ln.order}"/>'
            for ln in lines
        )
        groups.append(f'      <OrderedGroup ID="ro_{_esc(region)}">\n{refs}\n      </OrderedGroup>')
    return (
        '  <ReadingOrder>\n    <OrderedGroup ID="ro">\n'
        + "\n".join(groups)
        + "\n    </OrderedGroup>\n  </ReadingOrder>\n"
    )


def to_alto(labels: PageLabels) -> str:
    """ALTO v4 in the Riksarkivet/htrflow shape: a ``Processing`` provenance block, a ``ReadingOrder``,
    then ``TextBlock`` -> ``TextLine`` (``Shape``/``Polygon`` tight outline + ``BASELINE``) -> ``String``
    words. ``PC``/``WC`` confidences are 1.0 — this is ground truth, not a recognition guess."""
    regions = _ordered_regions(labels)
    blocks: list[str] = []
    for region, lines in regions:
        bx, by, bw, bh = _bbox([p for ln in lines for p in ln.polygon])
        tl: list[str] = []
        for ln in lines:
            lx, ly, lw, lh = _bbox(ln.polygon)
            baseline = f' BASELINE="{_pts(ln.baseline)}"' if ln.baseline else ""
            words = ln.words or [WordLabel(text=ln.text, polygon=ln.polygon)]
            strings: list[str] = []
            for wi, wd in enumerate(words):
                wx, wy, ww, wh = _bbox(wd.polygon)
                strings.append(
                    f'          <String ID="{_esc(region)}_l{ln.order}_w{wi}" HPOS="{wx}" '
                    f'VPOS="{wy}" WIDTH="{ww}" HEIGHT="{wh}" WC="1.0" CONTENT="{_esc(wd.text)}"/>'
                )
            tl.append(
                f'        <TextLine ID="{_esc(region)}_l{ln.order}" HPOS="{lx}" VPOS="{ly}" '
                f'WIDTH="{lw}" HEIGHT="{lh}"{baseline}>\n'
                f'          <Shape><Polygon POINTS="{_pts(ln.polygon)}"/></Shape>\n'
                + "\n".join(strings)
                + "\n        </TextLine>"
            )
        blocks.append(
            f'      <TextBlock ID="{_esc(region)}" HPOS="{bx}" VPOS="{by}" WIDTH="{bw}" '
            f'HEIGHT="{bh}" CS="false">\n' + "\n".join(tl) + "\n      </TextBlock>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<alto xmlns:xsi="{_ALTO_XSI}" xmlns="{_ALTO_NS}" xsi:schemaLocation="{_ALTO_SCHEMA}">\n'
        + _alto_description(_esc(labels.image))
        + _alto_reading_order(regions)
        + "  <Layout>\n"
        f'    <Page WIDTH="{labels.width}" HEIGHT="{labels.height}" PC="1.0" PHYSICAL_IMG_NR="0" ID="page">\n'
        "      <TopMargin/><LeftMargin/><RightMargin/><BottomMargin/>\n"
        "      <PrintSpace>\n"
        + "\n".join(blocks)
        + "\n      </PrintSpace>\n    </Page>\n  </Layout>\n</alto>\n"
    )


# THE label-format registry: name -> (exporter, file extension). Adding a format is ONE line here —
# EXPORTERS / EXTENSIONS are derived views (kept so existing imports work) and config.Fmt derives its
# StrEnum members from these names too.
FORMATS: dict[str, tuple[Callable[[PageLabels], str], str]] = {
    "json": (to_json, ".json"),
    "page": (to_page_xml, ".xml"),
    "alto": (to_alto, ".alto.xml"),
}
EXPORTERS: dict[str, Callable[[PageLabels], str]] = {name: fn for name, (fn, _ext) in FORMATS.items()}
EXTENSIONS: dict[str, str] = {name: ext for name, (_fn, ext) in FORMATS.items()}


def export(labels: PageLabels, fmt: str) -> str:
    """Serialize ``labels`` to one registered format (``json`` | ``page`` | ``alto``)."""
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown format {fmt!r}; options: {sorted(EXPORTERS)}")
    return EXPORTERS[fmt](labels)
