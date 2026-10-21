"""A crop's ground truth and its export formats.

:class:`PageLabels` is the label data (lines with their transcription, polygon, baseline, region, reading order and
tag); PAGE and ALTO XML are only serializations of it, each a ``PageLabels -> str`` function in :data:`EXPORTERS`.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Literal
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field

from diffu_page.layout.model import ROLE_NAMES, RUNNING_TEXT, TABLE

try:
    _VERSION = version("page-synth")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    _VERSION = "0"

type Point = tuple[int, int]
type LabelFormat = Literal["alto", "page"]


class LineLabel(BaseModel):
    """One text line: transcription, polygon, baseline polyline, region, reading-order index and tag."""

    text: str
    polygon: list[Point]
    baseline: list[Point] = Field(default_factory=list)  # left to right; empty if unknown
    region: str = RUNNING_TEXT  # a role's PAGE type (see ROLE_NAMES), "table" or "body"
    order: int = 0
    tag: str = ""  # "unclear" when the line read back badly


class PageLabels(BaseModel):
    """A page's full ground truth: the single value every exporter serializes."""

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


def _page_type(region: str) -> str:
    """A region's PAGE ``TextRegion@type``: a role's own type, a table cell's ``other``, running text's
    ``paragraph``."""
    if region in ROLE_NAMES:
        return region
    return "other" if region == TABLE else "paragraph"


def _ordered_regions(labels: PageLabels) -> list[tuple[str, list[LineLabel]]]:
    """Group lines by region, sort each region's lines by reading-order index, and return the regions
    themselves in reading order (by their first line's order). One source of truth for every exporter."""
    groups: dict[str, list[LineLabel]] = {}
    for line in labels.lines:
        groups.setdefault(line.region, []).append(line)
    ordered = {region: sorted(lines, key=lambda ln: ln.order) for region, lines in groups.items()}
    return sorted(ordered.items(), key=lambda kv: kv[1][0].order)


_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_PAGE_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_PAGE_SCHEMA = f"{_PAGE_NS} {_PAGE_NS}/pagecontent.xsd"
# Synthetic ground truth has no real authoring moment; a fixed UTC placeholder keeps the schema-required
# Metadata timestamps valid AND the export byte-reproducible for a given page.
_CREATED = "1970-01-01T00:00:00Z"


def _pts(points: list[Point]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _page_text_line(region: str, line: LineLabel) -> str:
    """One ``TextLine``: Coords, Baseline, TextEquiv, and the tag as Transkribus writes it into ``@custom``."""
    baseline = f'        <Baseline points="{_pts(line.baseline)}"/>\n' if line.baseline else ""
    custom = f' custom="structure {{type:{_esc(line.tag)};}}"' if line.tag else ""
    return (
        f'      <TextLine id="{_esc(region)}_l{line.order}" production="handwritten-cursive"{custom}>\n'
        f'        <Coords points="{_pts(line.polygon)}"/>\n'
        f"{baseline}"
        f"        <TextEquiv><Unicode>{_esc(line.text)}</Unicode></TextEquiv>\n"
        f"      </TextLine>"
    )


def to_page_xml(labels: PageLabels) -> str:
    """PRImA PAGE-XML (validates against the 2019 / 2013 content model): the required Metadata timestamps, an
    explicit ``ReadingOrder``, one typed ``TextRegion`` (Coords) per region, and a ``TextLine`` (Coords, Baseline,
    TextEquiv) per line."""
    regions = _ordered_regions(labels)
    refs = "\n".join(
        f'        <RegionRefIndexed index="{i}" regionRef="{_esc(region)}"/>' for i, (region, _) in enumerate(regions)
    )
    reading_order = (
        f'    <ReadingOrder>\n      <OrderedGroup id="ro">\n{refs}\n      </OrderedGroup>\n    </ReadingOrder>\n'
    )
    blocks: list[str] = []
    for region, lines in regions:
        rx, ry, rw, rh = _bbox([p for ln in lines for p in ln.polygon])
        rcoords = _pts([(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)])
        blocks.append(
            f'    <TextRegion id="{_esc(region)}" type="{_page_type(region)}">\n'
            f'      <Coords points="{rcoords}"/>\n'
            + "\n".join(_page_text_line(region, ln) for ln in lines)
            + "\n    </TextRegion>"
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
        "      <processingStepSettings>generator: diffu_page; labels: ground-truth</processingStepSettings>\n"
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
            f'        <ElementRef ID="ro_{_esc(region)}_l{ln.order}" REF="{_esc(region)}_l{ln.order}"/>' for ln in lines
        )
        groups.append(f'      <OrderedGroup ID="ro_{_esc(region)}">\n{refs}\n      </OrderedGroup>')
    return (
        '  <ReadingOrder>\n    <OrderedGroup ID="ro">\n'
        + "\n".join(groups)
        + "\n    </OrderedGroup>\n  </ReadingOrder>\n"
    )


def _alto_text_line(region: str, line: LineLabel) -> str:
    """One ``TextLine``: its box, its polygon as a ``Shape``, its baseline, and its text as one ``String``."""
    lx, ly, lw, lh = _bbox(line.polygon)
    baseline = f' BASELINE="{_pts(line.baseline)}"' if line.baseline else ""
    return (
        f'        <TextLine ID="{_esc(region)}_l{line.order}" HPOS="{lx}" VPOS="{ly}" '
        f'WIDTH="{lw}" HEIGHT="{lh}"{baseline}>\n'
        f'          <Shape><Polygon POINTS="{_pts(line.polygon)}"/></Shape>\n'
        f'          <String ID="{_esc(region)}_l{line.order}_s" HPOS="{lx}" VPOS="{ly}" WIDTH="{lw}" '
        f'HEIGHT="{lh}" WC="1.0" CONTENT="{_esc(line.text)}"/>\n'
        "        </TextLine>"
    )


def to_alto(labels: PageLabels) -> str:
    """ALTO v4 in the Riksarkivet/htrflow shape: a ``Processing`` provenance block, a ``ReadingOrder``,
    then ``TextBlock`` -> ``TextLine`` (``Shape``/``Polygon`` outline + ``BASELINE``) -> one ``String``.
    ``PC``/``WC`` confidences are 1.0 — this is ground truth, not a recognition guess."""
    regions = _ordered_regions(labels)
    blocks: list[str] = []
    for region, lines in regions:
        bx, by, bw, bh = _bbox([p for ln in lines for p in ln.polygon])
        blocks.append(
            f'      <TextBlock ID="{_esc(region)}" HPOS="{bx}" VPOS="{by}" WIDTH="{bw}" HEIGHT="{bh}" CS="false">\n'
            + "\n".join(_alto_text_line(region, ln) for ln in lines)
            + "\n      </TextBlock>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<alto xmlns:xsi="{_ALTO_XSI}" xmlns="{_ALTO_NS}" xsi:schemaLocation="{_ALTO_SCHEMA}">\n'
        + _alto_description(_esc(labels.image))
        + _alto_reading_order(regions)
        + "  <Layout>\n"
        f'    <Page WIDTH="{labels.width}" HEIGHT="{labels.height}" PC="1.0" PHYSICAL_IMG_NR="0" ID="page">\n'
        "      <TopMargin/><LeftMargin/><RightMargin/><BottomMargin/>\n"
        "      <PrintSpace>\n" + "\n".join(blocks) + "\n      </PrintSpace>\n    </Page>\n  </Layout>\n</alto>\n"
    )


EXPORTERS: dict[LabelFormat, Callable[[PageLabels], str]] = {"page": to_page_xml, "alto": to_alto}


def export(labels: PageLabels, fmt: LabelFormat) -> str:
    """Serialize ``labels`` to one format (``alto`` | ``page``).

    Raises:
        ValueError: If ``fmt`` is not a known format.
    """
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown format {fmt!r}; options: {sorted(EXPORTERS)}")
    return EXPORTERS[fmt](labels)
