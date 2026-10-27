"""A crop's ground truth and its export formats.

:class:`PageLabels` is the label data (lines with their transcription, polygon, baseline, region, reading order, tag,
production and table cell, and the ruled lines); PAGE and ALTO XML are only serializations of it, each a
``PageLabels -> str`` function in :data:`EXPORTERS`. PAGE writes a table as PAGE 2019 does: a ``TableRegion`` of nested
cell ``TextRegion``s, each with a ``TableCellRole``; ALTO, which has no tables, writes each cell as a ``TextBlock``.

The page readers' formats (:data:`READER_FILES`) are written from the same regions, as :class:`PageElements`: each text
region and each table one element, in reading order. :func:`to_markdown` is the page as Markdown (an element's lines one
under the other, a heading after ``## ``, a table as HTML), :func:`to_grounded_markdown` is DeepSeek-OCR 2's grounded
form of it (each element under its label and box), and the elements themselves, as JSON, are PaddleOCR-VL's layout
elements (each with its PP-DocLayoutV3 category, its box and the prompt and text its recogniser reads its crop with).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Self
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, computed_field, model_validator

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
    """One ``TableRegion``: the rows and columns the crop shows (a window into a page's table counts its own from 0,
    as PAGE's ``rows`` and ``rowIndex`` describe the table in this image), its box, and its cells as nested
    ``TextRegion``s."""
    refs = [cell.cell for cell in cells if cell.cell is not None]
    row0, column0 = min(ref.row for ref in refs), min(ref.column for ref in refs)
    rows = max(ref.row for ref in refs) - row0 + 1
    columns = max(ref.column for ref in refs) - column0 + 1
    rebased = [
        cell.model_copy(
            update={
                "cell": cell.cell.model_copy(update={"row": cell.cell.row - row0, "column": cell.cell.column - column0})
            }
        )
        for cell in cells
        if cell.cell is not None
    ]
    nested = "\n".join(_page_text_region(cell, indent="      ") for cell in rebased)
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


# PaddleOCR-VL's layout categories (PP-DocLayoutV3's label_list, its layout model from 1.5 on) for each kind of element.
# A catchword and a signature mark are page furniture under the text, as a footer is; the categories have neither.
LAYOUT_CATEGORIES: dict[LineKind, str] = {
    LineKind.BODY: "text",
    LineKind.TABLE: "table",
    LineKind.HEADING: "paragraph_title",
    LineKind.MARGINALIA: "aside_text",
    LineKind.PAGE_NUMBER: "number",
    LineKind.HEADER: "header",
    LineKind.FOOTER: "footer",
    LineKind.CATCH_WORD: "footer",
    LineKind.SIGNATURE_MARK: "footer",
}

# DeepSeek-OCR's labels for each kind of element, of those its paper's examples show: text, table and sub_title (a
# heading, written after "## "). They show none for page furniture or margin notes, so those are text.
GROUNDING_LABELS: dict[LineKind, str] = {
    LineKind.BODY: "text",
    LineKind.TABLE: "table",
    LineKind.HEADING: "sub_title",
    LineKind.MARGINALIA: "text",
    LineKind.PAGE_NUMBER: "text",
    LineKind.HEADER: "text",
    LineKind.FOOTER: "text",
    LineKind.CATCH_WORD: "text",
    LineKind.SIGNATURE_MARK: "text",
}

GROUNDING_SCALE = 999  # DeepSeek-OCR's boxes: on 0-999 of the image's width and height, as its decoder divides them

OCR_PROMPT = "OCR:"  # PaddleOCR-VL's recognition prompts: an element's crop read as text, or as a table (OTSL)
TABLE_PROMPT = "Table Recognition:"


class ElementLine(BaseModel):
    """One line of a page element: its text, its box (page px: x0, y0, x1, y1), how it was made, its tag (``unclear``
    when it read back badly) and, in a table, its cell, counted from the first row and column of the table in the
    image."""

    text: str
    box: Rect
    production: Production = Production.HANDWRITTEN
    tag: str = ""
    cell: CellRef | None = None


type TableGrid = list[list[str | None]]  # a table's cells, row by row: each its text, None where nothing is written


def _table_grid(lines: list[ElementLine]) -> TableGrid:
    """A table's cells by row and column, each its lines' texts one under the other."""
    texts: dict[tuple[int, int], list[str]] = {}
    for line in lines:
        if line.cell is not None:
            texts.setdefault((line.cell.row, line.cell.column), []).append(line.text)
    rows, columns = max(row for row, _ in texts) + 1, max(column for _, column in texts) + 1
    return [
        ["\n".join(texts[row, column]) if (row, column) in texts else None for column in range(columns)]
        for row in range(rows)
    ]


def _html_table(grid: TableGrid) -> str:
    """A table as the HTML DeepSeek-OCR writes in its Markdown: ``<table>``, ``<tr>`` and ``<td>`` alone, each cell's
    text escaped (``&``, ``<``, ``>``)."""
    rows = "".join("<tr>" + "".join(f"<td>{escape(text or '')}</td>" for text in row) + "</tr>" for row in grid)
    return f"<table>{rows}</table>"


def _otsl_table(grid: TableGrid) -> str:
    """A table as PaddleOCR-VL's recogniser reads it (OTSL): ``<fcel>`` and its text for a written cell, ``<ecel>``
    for an empty one, ``<nl>`` after each row."""
    return "".join("".join("<ecel>" if text is None else f"<fcel>{text}" for text in row) + "<nl>" for row in grid)


class PageElement(BaseModel):
    """One element of a page as a page reader reads it: a text region, or a whole table; its kind, its box (page px)
    and its lines in reading order (a table's cell by cell). As JSON it also gives its PaddleOCR-VL category, the
    prompt PaddleOCR-VL's recogniser reads its crop with and the text it reads there."""

    kind: LineKind
    box: Rect
    lines: list[ElementLine]

    @computed_field
    @property
    def category(self) -> str:
        """The element's layout category (PP-DocLayoutV3's)."""
        return LAYOUT_CATEGORIES[self.kind]

    @computed_field
    @property
    def task(self) -> str:
        """The prompt PaddleOCR-VL's recogniser reads the element's crop with."""
        return TABLE_PROMPT if self.kind is LineKind.TABLE else OCR_PROMPT

    @computed_field
    @property
    def text(self) -> str:
        """What the recogniser reads in the element's crop: a table as OTSL, any other element its lines one under
        the other."""
        if self.kind is LineKind.TABLE:
            return _otsl_table(_table_grid(self.lines))
        return "\n".join(line.text for line in self.lines)

    @property
    def markdown(self) -> str:
        """The element in the page's Markdown: a table as HTML, a heading after ``## `` on one line, any other element
        its lines one under the other (a soft line break each: still one paragraph)."""
        if self.kind is LineKind.TABLE:
            return _html_table(_table_grid(self.lines))
        if self.kind is LineKind.HEADING:
            return f"## {' '.join(line.text for line in self.lines)}"
        return "\n".join(line.text for line in self.lines)


class PageElements(BaseModel):
    """A page as the page readers read it (``<name>.elements.json``): its image, its size (px) and its elements in
    reading order."""

    image: str
    width: int
    height: int
    elements: list[PageElement]


def _element_line(line: LineLabel, row0: int, column0: int) -> ElementLine:
    """``line`` as an element's line, its cell counted from row ``row0`` and column ``column0`` of its table."""
    cell = (
        None
        if line.cell is None
        else line.cell.model_copy(update={"row": line.cell.row - row0, "column": line.cell.column - column0})
    )
    return ElementLine(text=line.text, box=_region_rect([line]), production=line.production, tag=line.tag, cell=cell)


def page_elements(labels: PageLabels) -> PageElements:
    """The page's elements in reading order: each text region one, each table one (its cells together, where its first
    cell is read), as PAGE and ALTO write their regions, so every format holds the same blocks."""
    elements: list[PageElement] = []
    for group in _top_level(_ordered_regions(labels)):
        lines = [line for region in group for line in region.lines]
        refs = [line.cell for line in lines if line.cell is not None]
        row0, column0 = min((ref.row for ref in refs), default=0), min((ref.column for ref in refs), default=0)
        elements.append(
            PageElement(
                kind=group[0].kind,
                box=_region_rect(lines),
                lines=[_element_line(line, row0, column0) for line in lines],
            )
        )
    return PageElements(image=labels.image, width=labels.width, height=labels.height, elements=elements)


def to_markdown(page: PageElements) -> str:
    """The page as Markdown: its elements in reading order, a blank line between two. Each line of the page stays a
    line of its own, as transcribed: Markdown reads lines one under the other as one paragraph."""
    return "\n\n".join(element.markdown for element in page.elements)


def _grounding(element: PageElement, page: PageElements) -> str:
    """DeepSeek-OCR's reference to an element: its label and its box, x on 0-999 of the page's width, y of its
    height."""
    x0, y0, x1, y1 = element.box
    sides = ((x0, page.width), (y0, page.height), (x1, page.width), (y1, page.height))
    box = ", ".join(str(round(GROUNDING_SCALE * value / size)) for value, size in sides)
    return f"<|ref|>{GROUNDING_LABELS[element.kind]}<|/ref|><|det|>[[{box}]]<|/det|>"


def to_grounded_markdown(page: PageElements) -> str:
    """The page as DeepSeek-OCR 2 writes a document grounded: :func:`to_markdown`'s elements, each under its label and
    box (``<|ref|>text<|/ref|><|det|>[[x0, y0, x1, y1]]<|/det|>``). Its decoder's clean-up (each reference taken out,
    three line breaks made two) gives :func:`to_markdown`'s text back."""
    return "\n\n".join(f"{_grounding(element, page)}\n{element.markdown}" for element in page.elements)


def to_elements_json(page: PageElements) -> str:
    """The page's elements as JSON: each with its PaddleOCR-VL category, box, recognition prompt, text and lines."""
    return page.model_dump_json(indent=1)


ELEMENTS_FILE = ".elements.json"  # the suffix of a crop's elements file, which the readers' sets are made from

# The page readers' labels of a crop, by the suffix of the file each is written to beside its image
READER_FILES: dict[str, Callable[[PageElements], str]] = {
    ".md": to_markdown,
    ".grounded.md": to_grounded_markdown,
    ELEMENTS_FILE: to_elements_json,
}
