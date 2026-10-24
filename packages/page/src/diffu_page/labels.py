"""A crop's ground truth and its export formats.

:class:`PageLabels` is the label data (lines with their transcription, polygon, baseline, region, reading order, tag,
production and table cell, and the ruled lines); PAGE and ALTO XML are only serializations of it, each a
``PageLabels -> str`` function in :data:`EXPORTERS`. PAGE writes a table as PAGE 2019 does: a ``TableRegion`` of nested
cell ``TextRegion``s, each with a ``TableCellRole``; ALTO, which has no tables, writes each cell as a ``TextBlock``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Self
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, model_validator

from diffu_page.layout.model import ROLE_NAMES, CellRef, LineKind, Production

try:
    _VERSION = version("diffu-page")
except PackageNotFoundError:  # not installed (e.g. running from a source checkout)
    _VERSION = "0"

type Point = tuple[int, int]
type Rect = tuple[int, int, int, int]  # x0, y0, x1, y1


class LabelFormat(StrEnum):
    """The formats a crop's labels are written in."""

    ALTO = "alto"
    PAGE = "page"


class LineLabel(BaseModel):
    """One text line: transcription, polygon, baseline polyline, region, source block, reading-order index and tag.

    The lines of one region and block form one ``TextRegion``/``TextBlock``: two columns of running text stay two."""

    text: str
    polygon: list[Point]
    baseline: list[Point] = Field(default_factory=list)  # left to right; empty if unknown
    region: LineKind = LineKind.BODY
    block: str = ""  # the template block the line came from; "" for labels made before blocks were kept
    order: int  # its place in reading order, which the exported ids are built from
    tag: str = ""  # "unclear" when the line read back badly
    production: Production = Production.HANDWRITTEN
    cell: CellRef | None = None  # the table cell the line is written in

    @model_validator(mode="after")
    def _table_lines_have_cells(self) -> Self:
        """PAGE says a line is in a table only through its cell, so a table line without one could not be written."""
        if (self.region is LineKind.TABLE) != (self.cell is not None):
            raise ValueError(f"a line is in the table region exactly when it has a table cell: {self.text!r}")
        return self


class LabelRegion(BaseModel):
    """One exported region: its id, unique in the page (the kind alone when it is the page's only region of that
    kind, else numbered in reading order), its kind, its lines in reading order and, for a table cell, the cell."""

    id: str
    kind: LineKind
    lines: list[LineLabel]
    cell: CellRef | None = None


class PageLabels(BaseModel):
    """A page's full ground truth: the single value every exporter serializes."""

    width: int
    height: int
    image: str = ""
    lines: list[LineLabel] = Field(default_factory=list)
    rules: list[Rect] = Field(default_factory=list)  # the ruled lines, page px


def _bbox(poly: list[Point]) -> tuple[int, int, int, int]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _esc(s: str) -> str:
    return escape(s, {'"': "&quot;"})  # values land in XML attributes, so quotes must be entities too


def _page_type(region: LineKind) -> str:
    """A region's PAGE ``TextRegion@type``: a role's own type, a table cell's ``other``, running text's
    ``paragraph``."""
    if region in ROLE_NAMES:
        return region.value
    return "other" if region is LineKind.TABLE else "paragraph"


def _ordered_regions(labels: PageLabels) -> list[LabelRegion]:
    """Group lines by region and block, sort each group's lines by reading-order index, and return the groups in
    reading order (by their first line's order). One source of truth for every exporter."""
    groups: dict[tuple[str, str], list[LineLabel]] = {}
    for line in labels.lines:
        groups.setdefault((line.region, line.block), []).append(line)
    ordered = sorted((sorted(lines, key=lambda ln: ln.order) for lines in groups.values()), key=lambda ls: ls[0].order)
    per_kind = Counter(lines[0].region for lines in ordered)
    numbered: Counter[str] = Counter()
    regions: list[LabelRegion] = []
    for lines in ordered:
        kind = lines[0].region
        numbered[kind] += 1
        region_id = kind if per_kind[kind] == 1 else f"{kind}_{numbered[kind]}"
        regions.append(LabelRegion(id=region_id, kind=kind, lines=lines, cell=lines[0].cell))
    return regions


def _rect_points(rect: Rect) -> str:
    x0, y0, x1, y1 = rect
    return _pts([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _region_rect(lines: list[LineLabel]) -> Rect:
    x, y, w, h = _bbox([p for line in lines for p in line.polygon])
    return x, y, x + w, y + h


_PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_PAGE_XSI = "http://www.w3.org/2001/XMLSchema-instance"
_PAGE_SCHEMA = f"{_PAGE_NS} {_PAGE_NS}/pagecontent.xsd"
# Synthetic ground truth has no real authoring moment; a fixed UTC placeholder keeps the schema-required
# Metadata timestamps valid AND the export byte-reproducible for a given page.
_CREATED = "1970-01-01T00:00:00Z"


def _pts(points: list[Point]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _page_text_line(region: str, line: LineLabel, index: int) -> str:
    """One ``TextLine``: Coords, Baseline, TextEquiv, and in ``@custom``, as Transkribus writes them, its place in its
    region and, for a tagged line, the tag over its whole text (``unclear {offset:0; length:n;}``)."""
    baseline = f'        <Baseline points="{_pts(line.baseline)}"/>\n' if line.baseline else ""
    tag = f" {_esc(line.tag)} {{offset:0; length:{len(line.text)};}}" if line.tag else ""
    production = "printed" if line.production is Production.PRINTED else "handwritten-cursive"
    return (
        f'      <TextLine id="{_esc(region)}_l{line.order}" production="{production}" '
        f'custom="readingOrder {{index:{index};}}{tag}">\n'
        f'        <Coords points="{_pts(line.polygon)}"/>\n'
        f"{baseline}"
        f"        <TextEquiv><Unicode>{_esc(line.text)}</Unicode></TextEquiv>\n"
        f"      </TextLine>"
    )


def _top_level(regions: list[LabelRegion]) -> list[list[LabelRegion]]:
    """The page's top-level regions in reading order: each text region on its own, a table's cells together, where
    its first cell is read."""
    groups: dict[str, list[LabelRegion]] = {}
    for region in regions:
        key = f"table {region.cell.table}" if region.cell is not None else f"region {region.id}"
        groups.setdefault(key, []).append(region)
    return list(groups.values())


def _page_text_region(region: LabelRegion, indent: str = "    ") -> str:
    """One ``TextRegion``: its type, its box, a ``TableCellRole`` when it is a table cell, and its lines."""
    role = (
        f'{indent}  <Roles><TableCellRole rowIndex="{region.cell.row}" columnIndex="{region.cell.column}"/></Roles>\n'
        if region.cell is not None
        else ""
    )
    return (
        f'{indent}<TextRegion id="{_esc(region.id)}" type="{_page_type(region.kind)}">\n'
        f'{indent}  <Coords points="{_rect_points(_region_rect(region.lines))}"/>\n'
        + role
        + "\n".join(_page_text_line(region.id, ln, k) for k, ln in enumerate(region.lines))
        + f"\n{indent}</TextRegion>"
    )


def _page_table(table_id: str, cells: list[LabelRegion]) -> str:
    """One ``TableRegion``: its rows and columns, its box, and its cells as nested ``TextRegion``s."""
    rows = len({cell.cell.row for cell in cells if cell.cell is not None})
    columns = len({cell.cell.column for cell in cells if cell.cell is not None})
    nested = "\n".join(_page_text_region(cell, indent="      ") for cell in cells)
    return (
        f'    <TableRegion id="{_esc(table_id)}" rows="{rows}" columns="{columns}">\n'
        f'      <Coords points="{_rect_points(_region_rect([line for cell in cells for line in cell.lines]))}"/>\n'
        f"{nested}\n    </TableRegion>"
    )


def to_page_xml(labels: PageLabels) -> str:
    """PRImA PAGE-XML (validates against the 2019 content model): the required Metadata timestamps, an explicit
    ``ReadingOrder``, one typed ``TextRegion`` (Coords) per region and one ``TableRegion`` per table (its cells
    nested ``TextRegion``s with a ``TableCellRole``), a ``TextLine`` (Coords, Baseline, TextEquiv, production) per
    line, and a ``SeparatorRegion`` per ruled line."""
    top = _top_level(_ordered_regions(labels))
    tables = [group for group in top if group[0].cell is not None]
    table_ids = {
        id(group[0]): "table_region" if len(tables) == 1 else f"table_region_{k}" for k, group in enumerate(tables, 1)
    }
    ids = [table_ids.get(id(group[0]), group[0].id) for group in top]
    refs = "\n".join(f'        <RegionRefIndexed index="{i}" regionRef="{_esc(ref)}"/>' for i, ref in enumerate(ids))
    reading_order = (
        f'    <ReadingOrder>\n      <OrderedGroup id="ro">\n{refs}\n      </OrderedGroup>\n    </ReadingOrder>\n'
    )
    blocks = [
        _page_table(ref, group) if group[0].cell is not None else _page_text_region(group[0])
        for ref, group in zip(ids, top, strict=True)
    ]
    blocks += [
        f'    <SeparatorRegion id="rule_{k}">\n      <Coords points="{_rect_points(rect)}"/>\n    </SeparatorRegion>'
        for k, rect in enumerate(labels.rules, 1)
    ]
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


def _region_tag(kind: str) -> str:
    return f"region_{kind}"


def _production_tag(production: Production) -> str:
    return f"production_{production.value}"


def _readback_tag(tag: str) -> str:
    return f"readback_{tag}"


def _alto_tags(regions: list[LabelRegion]) -> str:
    """ALTO ``<Tags>``: what PAGE says in region types, ``production`` and ``@custom``, as the tags the blocks and
    lines reference (``TAGREFS``): a ``StructureTag`` per region kind, an ``OtherTag`` per production and per
    read-back tag (``unclear``). Only the tags in use, so every one is referenced."""
    lines = [line for region in regions for line in region.lines]
    kinds = sorted({region.kind for region in regions})
    productions = sorted({line.production for line in lines})
    readbacks = sorted({line.tag for line in lines if line.tag})
    tags = [
        *(f'    <StructureTag ID="{_esc(_region_tag(kind))}" TYPE="region" LABEL="{_esc(kind)}"/>' for kind in kinds),
        *(
            f'    <OtherTag ID="{_production_tag(production)}" TYPE="production" LABEL="{production.value}"/>'
            for production in productions
        ),
        *(
            f'    <OtherTag ID="{_esc(_readback_tag(tag))}" TYPE="readback" LABEL="{_esc(tag)}" '
            'DESCRIPTION="the line read back badly: the drawing may not show its text"/>'
            for tag in readbacks
        ),
    ]
    return "  <Tags>\n" + "\n".join(tags) + "\n  </Tags>\n"


def _alto_reading_order(regions: list[LabelRegion]) -> str:
    """ALTO ``<ReadingOrder>``: a nested ``OrderedGroup`` per region, an ``ElementRef`` per line (REF =
    the line's ``TextLine`` ID) — the htrflow shape, so a reader can recover line sequence without geometry."""
    groups: list[str] = []
    for region in regions:
        refs = "\n".join(
            f'        <ElementRef ID="ro_{_esc(region.id)}_l{ln.order}" REF="{_esc(region.id)}_l{ln.order}"/>'
            for ln in region.lines
        )
        groups.append(f'      <OrderedGroup ID="ro_{_esc(region.id)}">\n{refs}\n      </OrderedGroup>')
    return (
        '  <ReadingOrder>\n    <OrderedGroup ID="ro">\n'
        + "\n".join(groups)
        + "\n    </OrderedGroup>\n  </ReadingOrder>\n"
    )


def _alto_text_line(region: str, line: LineLabel) -> str:
    """One ``TextLine``: its box, its polygon as a ``Shape``, its baseline, and its text as one ``String``."""
    lx, ly, lw, lh = _bbox(line.polygon)
    baseline = f' BASELINE="{_pts(line.baseline)}"' if line.baseline else ""
    tags = " ".join([_production_tag(line.production), *([_readback_tag(line.tag)] if line.tag else [])])
    return (
        f'        <TextLine ID="{_esc(region)}_l{line.order}" HPOS="{lx}" VPOS="{ly}" '
        f'WIDTH="{lw}" HEIGHT="{lh}"{baseline} TAGREFS="{_esc(tags)}">\n'
        f'          <Shape><Polygon POINTS="{_pts(line.polygon)}"/></Shape>\n'
        f'          <String ID="{_esc(region)}_l{line.order}_s" HPOS="{lx}" VPOS="{ly}" WIDTH="{lw}" '
        f'HEIGHT="{lh}" WC="1.0" CONTENT="{_esc(line.text)}"/>\n'
        "        </TextLine>"
    )


def to_alto(labels: PageLabels) -> str:
    """ALTO v4 in the Riksarkivet/htrflow shape: a ``Processing`` provenance block, the ``Tags`` that carry what
    PAGE says in region types, production and Transkribus tags, a ``ReadingOrder``, then ``TextBlock`` (its region
    kind in ``TAGREFS``) -> ``TextLine`` (``Shape``/``Polygon`` outline + ``BASELINE``, its production and read-back
    tag in ``TAGREFS``) -> one ``String``, a table cell's lines in their own ``TextBlock``, and a ``GraphicalElement``
    per ruled line.
    ``PC``/``WC`` confidences are 1.0 — this is ground truth, not a recognition guess."""
    regions = _ordered_regions(labels)
    blocks: list[str] = []
    for region in regions:
        bx, by, bw, bh = _bbox([p for ln in region.lines for p in ln.polygon])
        blocks.append(
            f'      <TextBlock ID="{_esc(region.id)}" HPOS="{bx}" VPOS="{by}" WIDTH="{bw}" HEIGHT="{bh}" CS="false" '
            f'TAGREFS="{_esc(_region_tag(region.kind))}">\n'
            + "\n".join(_alto_text_line(region.id, ln) for ln in region.lines)
            + "\n      </TextBlock>"
        )
    blocks += [
        f'      <GraphicalElement ID="rule_{k}" HPOS="{x0}" VPOS="{y0}" WIDTH="{x1 - x0}" HEIGHT="{y1 - y0}"/>'
        for k, (x0, y0, x1, y1) in enumerate(labels.rules, 1)
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<alto xmlns:xsi="{_ALTO_XSI}" xmlns="{_ALTO_NS}" xsi:schemaLocation="{_ALTO_SCHEMA}">\n'
        + _alto_description(_esc(labels.image))
        + _alto_tags(regions)
        + _alto_reading_order(regions)
        + "  <Layout>\n"
        f'    <Page WIDTH="{labels.width}" HEIGHT="{labels.height}" PC="1.0" PHYSICAL_IMG_NR="0" ID="page">\n'
        "      <TopMargin/><LeftMargin/><RightMargin/><BottomMargin/>\n"
        "      <PrintSpace>\n" + "\n".join(blocks) + "\n      </PrintSpace>\n    </Page>\n  </Layout>\n</alto>\n"
    )


EXPORTERS: dict[LabelFormat, Callable[[PageLabels], str]] = {LabelFormat.PAGE: to_page_xml, LabelFormat.ALTO: to_alto}


def export(labels: PageLabels, fmt: LabelFormat) -> str:
    """Serialize ``labels`` to one format (``alto`` | ``page``).

    Raises:
        ValueError: If ``fmt`` is not a known format.
    """
    if fmt not in EXPORTERS:
        raise ValueError(f"unknown format {fmt!r}; options: {sorted(EXPORTERS)}")
    return EXPORTERS[fmt](labels)
