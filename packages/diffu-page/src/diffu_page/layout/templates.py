"""Page templates — every synthetic page format, its vocab constants, and :func:`build_layout`.

Each template carves the body rect into regions and flows text into line boxes; all reduce to the
same :class:`~diffu_page.layout.model.LineBox` seam the renderer/compositor consume. ``TEMPLATES``
maps a :class:`PageSpec.template` name to its builder; :func:`build_layout` dispatches and applies
the optional manuscript-edit passes.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal, Protocol

from .edits import _apply_drop_cap, _apply_edits
from .flow import _box, _fit_to_width, _flow, _flow_paragraphs, _page_elements, _wrap
from .model import Align, Column, Layout, LineBox, Measure, PageSpec, Rule, TableColumn

type LeafPrefer = Literal["denser", "blank", "left", "right"]  # which leaf to_single_leaf keeps


def _single_column(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    extra, body = _page_elements(spec, rng)
    flowed = _flow_paragraphs(
        paragraphs, body, spec, rng, "body", measure, margin_keys=spec.margin_keys or None
    )
    return Layout(spec=spec, boxes=extra + flowed)


def _multi_column(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    n, gut = max(2, spec.n_columns), spec.gutter
    min_col = max(spec.line_height, int(spec.px_per_char * 4))  # a column must fit ~4 chars
    col_w = ((bx1 - bx0) - (n - 1) * gut) // n
    while n > 1 and col_w < min_col:  # guard: too many columns for the width -> drop columns
        n -= 1
        col_w = ((bx1 - bx0) - (n - 1) * gut) // n
    step = int(spec.line_height * (1.0 + spec.line_gap))
    capacity = max(1, (by1 - by0) // step)  # lines per column

    all_lines = _wrap(paragraphs, col_w, spec, measure)
    out: list[LineBox] = list(extra)
    rules: list[Rule] = []
    for c in range(n):
        x0 = bx0 + c * (col_w + gut)
        chunk = all_lines[c * capacity : (c + 1) * capacity]
        if not chunk:
            break
        col_boxes, _ = _flow(chunk, (x0, by0, x0 + col_w, by1), spec, rng, f"col{c}")
        out.extend(col_boxes)
        if c < n - 1:  # a separator rule sits in the gutter after this column
            gx = x0 + col_w + gut // 2
            rules.append(Rule(x1=gx, y1=by0, x2=gx, y2=by1, width=2))
    return Layout(spec=spec, boxes=out, rules=rules)


_MONTHS = [
    "Jan",
    "Febr",
    "Martii",
    "April",
    "Maii",
    "Junii",
    "Julii",
    "Aug",
    "Sept",
    "Octob",
    "Novemb",
    "Decemb",
]
# A realistic pre-printed register form: varied column widths + typed cells + right-aligned money columns.
_DEFAULT_TABLE_COLUMNS = [
    TableColumn(title="N:o", weight=0.6, kind="rownum", align="center"),
    TableColumn(title="Namn", weight=2.4, kind="name"),
    TableColumn(title="Hemvist", weight=1.8, kind="place"),
    TableColumn(title="Datum", weight=1.3, kind="date"),
    TableColumn(title="Anteckning", weight=2.2, kind="text"),
    TableColumn(title="Dlr", weight=0.8, kind="money", align="right"),
    TableColumn(title="ore", weight=0.7, kind="money", align="right"),
]


class _CellFn(Protocol):
    """Appends one fitted cell box at column ``ci`` / row top ``y`` (see :func:`_cell_appender`)."""

    def __call__(
        self, text: str, ci: int, y: int, region: str, align: Align, printed: bool = False
    ) -> None: ...


def _column_edges(columns: list[TableColumn], bx0: int, grid_w: int) -> list[int]:
    """Column x-boundaries, proportioned by weight."""
    total_w = sum(c.weight for c in columns)
    edges, acc = [bx0], 0.0
    for c in columns:
        acc += c.weight
        edges.append(bx0 + int(grid_w * acc / total_w))
    return edges


def _fragment_source(paragraphs: list[str]) -> Callable[[int], str]:
    """A ``frag(n)`` closure cycling ``n``-word fragments out of ``paragraphs`` for typed text cells."""
    tokens = (" ".join(paragraphs)).split() or ["hemman"]
    ti = 0

    def frag(n: int) -> str:
        nonlocal ti
        out = " ".join(tokens[ti : ti + n]) or tokens[ti % len(tokens)]
        ti = (ti + n) % max(1, len(tokens))
        return out

    return frag


def _cell_text(kind: str, r: int, rng: random.Random, frag: Callable[[int], str]) -> str:
    """Typed content for one cell: a row number, a date, a money figure, or a text fragment."""
    if kind == "rownum":
        return f"{r}."
    if kind == "date":
        return f"d. {rng.randint(1, 28)} {rng.choice(_MONTHS)}"
    if kind == "money":
        return str(rng.randint(1, 200))
    return frag(2 if kind == "name" else 1 if kind == "place" else rng.randint(1, 3))


def _cell_appender(boxes: list[LineBox], edges: list[int], spec: PageSpec, rng: random.Random) -> _CellFn:
    """The ``cell(...)`` closure the grid builders share: fit the text to its column and append its box."""

    def cell(text: str, ci: int, y: int, region: str, align: Align, printed: bool = False) -> None:
        cx0, cx1 = edges[ci], edges[ci + 1]
        w = cx1 - cx0 - 12
        boxes.append(
            _box(
                _fit_to_width(text, w, spec),
                cx0 + 6,
                y + 4,
                w,
                spec.line_height,
                spec,
                rng,
                region,
                align,
                printed,
            )
        )

    return cell


def _semi_table_header_row(
    columns: list[TableColumn],
    by0: int,
    bx0: int,
    edges: list[int],
    spec: PageSpec,
    rules: list[Rule],
    cell: _CellFn,
) -> None:
    """The printed header row: a top rule + one titled cell per column."""
    rules.append(Rule(x1=bx0, y1=by0, x2=edges[-1], y2=by0, width=1))
    for ci, col in enumerate(columns):  # printed header row
        cell(col.title, ci, by0, "table_header", "center", printed=spec.typeset_furniture)


def _semi_table_body_rows(
    columns: list[TableColumn],
    rows: int,
    row_h: int,
    by0: int,
    bx0: int,
    edges: list[int],
    spec: PageSpec,
    rng: random.Random,
    frag: Callable[[int], str],
    rules: list[Rule],
    cell: _CellFn,
) -> None:
    """The typed body rows: a row rule + one kind-typed cell per column, per row."""
    for r in range(1, rows):  # typed body rows
        cy = by0 + r * row_h
        rules.append(Rule(x1=bx0, y1=cy, x2=edges[-1], y2=cy, width=1))
        for ci, col in enumerate(columns):
            figure = col.kind in ("rownum", "money")
            printed = spec.typeset_furniture and figure  # pre-printed forms only; else handwrite the figures
            cell(
                _cell_text(col.kind, r, rng, frag),
                ci,
                cy,
                "row_number" if figure else "table",
                col.align,
                printed,
            )


def _semi_table_summa_row(
    columns: list[TableColumn],
    total_y: int,
    bx0: int,
    edges: list[int],
    spec: PageSpec,
    rng: random.Random,
    rules: list[Rule],
    cell: _CellFn,
) -> None:
    """A 'Summa' totals row: a heavy rule, the label in the first text column, right-aligned money totals."""
    rules.append(Rule(x1=bx0, y1=total_y, x2=edges[-1], y2=total_y, width=2))
    text_ci = next((i for i, c in enumerate(columns) if c.kind in ("text", "name")), 0)
    cell("Summa", text_ci, total_y, "table_header", "right", printed=spec.typeset_furniture)
    for ci, col in enumerate(columns):
        if col.kind == "money":
            cell(str(rng.randint(200, 900)), ci, total_y, "table", "right", printed=spec.typeset_furniture)


def _semi_table(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A TYPED ruled register/ledger grid: proportioned columns each with a content KIND (row number, name,
    place, date, free text, or a right-aligned money figure), a printed header row, and a ``Summa`` totals
    row — the pre-printed register forms in the corpus (varied widths + numeric columns), not a uniform grid
    of random fragments. ``spec.table_cols`` selects how many of the default columns to use."""
    extra, (bx0, by0, bx1, by1) = _page_elements(spec, rng)
    columns = _DEFAULT_TABLE_COLUMNS[: max(2, spec.table_cols)]
    grid_h = by1 - by0
    edges = _column_edges(columns, bx0, bx1 - bx0)  # column x-boundaries, proportioned by weight
    has_money = any(c.kind == "money" for c in columns)
    row_h = max(spec.line_height + 8, grid_h // max(1, spec.table_rows))
    rows = min(spec.table_rows, max(1, grid_h // row_h - (1 if has_money else 0)))  # reserve a Summa row
    frag = _fragment_source(paragraphs)
    boxes: list[LineBox] = list(extra)
    rules: list[Rule] = []
    cell = _cell_appender(boxes, edges, spec, rng)

    _semi_table_header_row(columns, by0, bx0, edges, spec, rules, cell)
    _semi_table_body_rows(columns, rows, row_h, by0, bx0, edges, spec, rng, frag, rules, cell)
    rows_drawn = rows
    total_y = by0 + rows * row_h
    if has_money and total_y + row_h <= by1:  # a 'Summa' totals row with right-aligned column totals
        _semi_table_summa_row(columns, total_y, bx0, edges, spec, rng, rules, cell)
        rows_drawn += 1

    for cx in edges:  # column rules (incl. outer edges)
        rules.append(Rule(x1=cx, y1=by0, x2=cx, y2=by0 + rows_drawn * row_h, width=1))
    return Layout(spec=spec, boxes=boxes, rules=rules)


def _columns(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """Borderless columns at arbitrary positions/widths/alignments (spec.columns) — no rules."""
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    cols = spec.columns or [Column(x_frac=0.0, w_frac=1.0)]
    span = bx1 - bx0
    out: list[LineBox] = list(extra)
    chunks = [paragraphs[i :: len(cols)] for i in range(len(cols))]  # round-robin split across columns
    for col, chunk in zip(cols, chunks, strict=False):
        cx0 = bx0 + int(col.x_frac * span)
        cw = max(spec.line_height, int(col.w_frac * span))
        col_boxes, _ = _flow(
            _wrap(chunk, cw, spec, measure), (cx0, by0, cx0 + cw, by1), spec, rng, "col", col.align
        )
        out.extend(col_boxes)
    return Layout(spec=spec, boxes=out)


def _key_value(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """Borderless label/value rows (registers, forms) — one paragraph per row, split on ``kv_delim``."""
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    lw = int(spec.kv_label_frac * (bx1 - bx0))
    vx = bx0 + lw + spec.gutter // 2
    step = int(spec.line_height * (1.0 + spec.line_gap))
    out: list[LineBox] = list(extra)
    y = by0
    for para in paragraphs:
        if y + spec.line_height > by1:
            break
        key, sep, val = para.partition(spec.kv_delim)
        if sep:
            out.append(_box(key.strip() + sep, bx0, y, lw, spec.line_height, spec, rng, "kv_label"))
            out.append(_box(val.strip(), vx, y, bx1 - vx, spec.line_height, spec, rng, "kv_value"))
        else:
            out.append(_box(para.strip(), bx0, y, bx1 - bx0, spec.line_height, spec, rng, "kv_value"))
        y += step
    return Layout(spec=spec, boxes=out)


def _spread_leaf(
    paragraphs: list[str],
    rect: tuple[int, int, int, int],
    spec: PageSpec,
    rng: random.Random,
    region: str,
    *,
    page_number: str,
    pageno_align: Align,
    measure: Measure | None,
) -> list[LineBox]:
    """One leaf of a two-page spread: a corner page number + body flowed into ``rect``."""
    x0, x1 = rect[0], rect[2]
    pw = max(spec.line_height, int(len(page_number) * spec.px_per_char * 1.4))
    px = x0 if pageno_align == "left" else x1 - pw
    boxes = [
        _box(
            page_number,
            px,
            int(spec.margin * 0.4),
            pw,
            int(spec.line_height * 1.1),
            spec,
            rng,
            "page_number",
            pageno_align,
        )
    ]
    boxes.extend(_flow_paragraphs(paragraphs, rect, spec, rng, region, measure))
    return boxes


def _spread(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """An open-book opening: two leaves separated by a central gutter, page numbers in the OUTER corners,
    and (occasionally) one blank leaf — the dominant geometry of real bound-volume scans."""
    m, w, h = spec.margin, spec.width, spec.height
    cx = w // 2
    g = max(spec.line_height, int(0.03 * w))  # half the central no-text gutter
    top, bot = m + int(spec.line_height * 1.4), h - m  # room for the corner page numbers
    half = max(1, len(paragraphs) // 2)
    base = rng.randint(2, 400) * 2  # an even verso number on the left; recto = +1 on the right
    blank = rng.choice(("left", "right")) if rng.random() < 0.15 else None
    boxes: list[LineBox] = []
    if blank != "left":
        boxes += _spread_leaf(
            paragraphs[:half],
            (m, top, cx - g, bot),
            spec,
            rng,
            "left",
            page_number=str(base),
            pageno_align="left",
            measure=measure,
        )
    if blank != "right":
        boxes += _spread_leaf(
            paragraphs[half:],
            (cx + g, top, w - m, bot),
            spec,
            rng,
            "right",
            page_number=str(base + 1),
            pageno_align="right",
            measure=measure,
        )
    return Layout(spec=spec, boxes=boxes)


def _letter(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A letter / sparse leaf: a short block of body text in the TOP region, a large blank lower area, and
    a right-aligned closing + signature near the bottom. Matches the many sparse single pages in the corpus
    (letters, notes, endorsements) that a dense full-column template never produces — so the density
    distribution reaches its sparse tail, not just full pages."""
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    top_h = int((by1 - by0) * rng.uniform(0.32, 0.58))  # body fills only the top; the rest stays blank
    boxes = list(extra) + _flow_paragraphs(
        paragraphs, (bx0, by0, bx1, by0 + top_h), spec, rng, "body", measure
    )
    closing = rng.choice(["Hogaktningsfullt", "Vordsammast", "Odmjukast", "Eder tillgifne"])
    name = rng.choice(["A. Lindqvist", "C. J. Bergstrom", "Joh. Andersson", "E. Sundberg", "P. Wahlstrom"])
    sh = spec.line_height
    half = (bx1 - bx0) // 2
    sy = by1 - int(sh * 3.2)  # sit the closing well down the page, in the blank lower area
    cw = min(half, max(sh, int(len(closing) * spec.px_per_char)))
    nw = min(half, max(sh, int(len(name) * spec.px_per_char)))
    boxes.append(_box(closing, bx1 - cw, sy, cw, sh, spec, rng, "signature", "right"))
    boxes.append(_box(name, bx1 - nw, sy + int(sh * 1.5), nw, sh, spec, rng, "signature", "right"))
    return Layout(spec=spec, boxes=boxes)


def _register(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A leader-dot register: section headers + ``label <leader> value`` entries with the value right-aligned
    in a narrow right column and a thin leader rule bridging the gap. Matches the parish / tax registers that
    dominate the corpus (a name list with a number against each entry, grouped under headers).

    Paragraphs split on ``kv_delim`` become entries (label, value); paragraphs WITHOUT the delimiter become
    centred section headers. The leader is a ``Rule`` on the baseline — reusing the one furniture primitive.
    """
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    val_w = max(2 * spec.line_height, int(0.14 * (bx1 - bx0)))  # narrow right-hand value column
    vx = bx1 - val_w
    step = int(spec.line_height * (1.0 + spec.line_gap))
    boxes: list[LineBox] = list(extra)
    rules: list[Rule] = []
    y = by0
    for para in paragraphs:
        if y + spec.line_height > by1:
            break
        label, sep, value = para.partition(spec.kv_delim)
        if not sep or not value.strip():  # a section header — centred, no leader
            hw = min(bx1 - bx0, max(spec.line_height, int(len(para) * spec.px_per_char)))
            boxes.append(
                _box(
                    para.strip(),
                    bx0 + ((bx1 - bx0) - hw) // 2,
                    y,
                    hw,
                    spec.line_height,
                    spec,
                    rng,
                    "reg_header",
                    "center",
                )
            )
            y += step
            continue
        lw = min(vx - bx0 - spec.line_height, max(spec.line_height, int(len(label) * spec.px_per_char)))
        boxes.append(_box(label.strip(), bx0, y, lw, spec.line_height, spec, rng, "reg_label"))
        boxes.append(_box(value.strip(), vx, y, val_w, spec.line_height, spec, rng, "reg_value", "right"))
        ly = y + int(spec.line_height * spec.baseline_frac)  # leader sits on the baseline
        rules.append(Rule(x1=bx0 + lw + 10, y1=ly, x2=vx - 10, y2=ly, width=1, style="dashed"))
        y += step
    return Layout(spec=spec, boxes=boxes, rules=rules)


# Ledger number-column headers, cycled left to right.
_LEDGER_HEADERS = ["Dlr", "ore", "Krono", "Skatte"]


def _ledger_num_x(bx1: int, n_num: int, num_w: int, i: int) -> int:
    """Left edge of numeric column ``i`` (columns hang off the right page edge)."""
    return bx1 - (n_num - i) * num_w


def _ledger_header_row(
    spec: PageSpec,
    rng: random.Random,
    boxes: list[LineBox],
    rules: list[Rule],
    *,
    bx0: int,
    bx1: int,
    by0: int,
    desc_x1: int,
    n_num: int,
    num_w: int,
    step: int,
) -> None:
    """The typeset header row: 'Anteckning' over the description column, cycled number-column titles,
    and the heavy rule under the row."""
    boxes.append(
        _box(
            "Anteckning",
            bx0 + 6,
            by0,
            desc_x1 - bx0 - 12,
            spec.line_height,
            spec,
            rng,
            "table_header",
            "left",
            printed=spec.typeset_furniture,
        )
    )
    for i in range(n_num):
        boxes.append(
            _box(
                _LEDGER_HEADERS[i % len(_LEDGER_HEADERS)],
                _ledger_num_x(bx1, n_num, num_w, i) + 4,
                by0,
                num_w - 8,
                spec.line_height,
                spec,
                rng,
                "table_header",
                "center",
                printed=spec.typeset_furniture,
            )
        )
    rules.append(Rule(x1=bx0, y1=by0 + step - 4, x2=bx1, y2=by0 + step - 4, width=2))


def _ledger_summa_row(
    spec: PageSpec,
    rng: random.Random,
    boxes: list[LineBox],
    rules: list[Rule],
    *,
    y: int,
    bx0: int,
    bx1: int,
    desc_x1: int,
    n_num: int,
    num_w: int,
) -> None:
    """A subtotal row: a rule above it, then a right-aligned Summa + a total per number column."""
    rules.append(Rule(x1=bx0, y1=y - 2, x2=bx1, y2=y - 2, width=1))
    boxes.append(
        _box(
            "Summa",
            desc_x1 - 6 * spec.line_height,
            y,
            5 * spec.line_height,
            spec.line_height,
            spec,
            rng,
            "table",
            "right",
        )
    )
    for i in range(n_num):
        boxes.append(
            _box(
                str(rng.randint(40, 220)),
                _ledger_num_x(bx1, n_num, num_w, i) + 4,
                y,
                num_w - 8,
                spec.line_height,
                spec,
                rng,
                "table",
                "right",
            )
        )


def _ledger_rows(
    spec: PageSpec,
    rng: random.Random,
    tokens: list[str],
    boxes: list[LineBox],
    rules: list[Rule],
    *,
    bx0: int,
    bx1: int,
    by0: int,
    by1: int,
    desc_x1: int,
    n_num: int,
    num_w: int,
    step: int,
) -> None:
    """The data rows: a description + sparse number cells per row, with a Summa subtotal after each group."""
    y, ti, group, span = by0 + step, 0, 0, rng.randint(5, 8)
    while y + 2 * spec.line_height <= by1 and ti < len(tokens):
        desc = " ".join(tokens[ti : ti + rng.randint(2, 4)])
        ti += 3
        boxes.append(_box(desc, bx0 + 6, y, desc_x1 - bx0 - 12, spec.line_height, spec, rng, "table"))
        for i in range(n_num):
            if rng.random() < 0.8:
                boxes.append(
                    _box(
                        str(rng.randint(1, 48)),
                        _ledger_num_x(bx1, n_num, num_w, i) + 4,
                        y,
                        num_w - 8,
                        spec.line_height,
                        spec,
                        rng,
                        "table",
                        "right",
                    )
                )
        y += step
        group += 1
        if group % span == 0:  # a subtotal row: a rule above it, then a right-aligned Summa
            _ledger_summa_row(
                spec, rng, boxes, rules, y=y, bx0=bx0, bx1=bx1, desc_x1=desc_x1, n_num=n_num, num_w=num_w
            )
            y += step
            span = rng.randint(5, 8)


def _ledger(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A numeric ledger: a wide description column + N narrow right-aligned number columns divided by vertical
    rules, a typeset header row, and a subtotal row (rule + ``Summa``) after each group. Matches the
    tenant/tax ledgers with money columns (Dlr / ore) and running totals.
    """
    extra, body = _page_elements(spec, rng)
    bx0, by0, bx1, by1 = body
    n_num = max(1, min(4, spec.table_cols - 1))  # narrow numeric columns on the right
    num_w = max(spec.line_height, int(0.09 * (bx1 - bx0)))
    desc_x1 = bx1 - n_num * num_w  # right edge of the wide description column
    step = int(spec.line_height * (1.0 + spec.line_gap))
    tokens = (" ".join(paragraphs)).split() or ["hemman"]
    boxes: list[LineBox] = list(extra)
    rules: list[Rule] = [
        Rule(x1=bx1 - i * num_w, y1=by0, x2=bx1 - i * num_w, y2=by1, width=1) for i in range(n_num + 1)
    ]
    _ledger_header_row(
        spec,
        rng,
        boxes,
        rules,
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        desc_x1=desc_x1,
        n_num=n_num,
        num_w=num_w,
        step=step,
    )
    _ledger_rows(
        spec,
        rng,
        tokens,
        boxes,
        rules,
        bx0=bx0,
        bx1=bx1,
        by0=by0,
        by1=by1,
        desc_x1=desc_x1,
        n_num=n_num,
        num_w=num_w,
        step=step,
    )
    return Layout(spec=spec, boxes=boxes, rules=rules)


def _blank_leaf(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A near-blank leaf: page furniture + only a few stray lines (or none), the rest empty substrate. The
    paper texture + verso bleed-through become the dominant content, so the model learns that empty regions
    are valid rather than hallucinating text. Matches ``tomma_sidor`` and the blank half of many spreads."""
    extra, (bx0, by0, bx1, by1) = _page_elements(spec, rng)
    boxes = list(extra)
    n = rng.randint(0, 4)  # a handful of stray lines, or a truly blank leaf
    if n and paragraphs:
        y = by0 + rng.randint(0, max(1, (by1 - by0) // 3))  # start somewhere in the top third
        step = int(spec.line_height * (1.0 + spec.line_gap))
        for line in _wrap(paragraphs, bx1 - bx0, spec, measure)[:n]:
            if y + spec.line_height > by1:
                break
            boxes.append(_box(line, bx0, y, bx1 - bx0, spec.line_height, spec, rng, "body"))
            y += step
    return Layout(spec=spec, boxes=boxes)


# Boskapslängd / landskapshandling vocabulary (16th-17th c. Swedish tax & cattle rolls): parish/tenant
# section names + the dues (livestock, grain, money) each entry lists against a small figure.
_LANDSKAP_PLACES = [
    "Vreta", "Lüng", "Klista", "Kärna", "Ledberg", "Rystadh", "Örtomta", "Askeby",
    "Bankeberg", "Sjögestad", "Flistadh", "Kaga", "Wreta Kloster", "Törnevalla",
]  # fmt: skip
_LANDSKAP_KINDS = [
    "Sochn", "Bönder", "Skatte Bönder", "Fralse Bönder", "Kloster Bönder",
    "Cronohemman", "Ödegårdar", "Landbönder",
]  # fmt: skip
_LANDSKAP_DUES = [
    "Penningar", "Spannemåhl", "Korn", "Rågh", "Humbla", "Smör", "Oxar", "Koor",
    "Ungnöt", "Får", "Swin", "Getter", "Häst", "Stodh", "Fåhl", "Dagswerken",
    "Nötboskap", "Höns", "Ålägenhet", "Tijonde",
]  # fmt: skip


def _landskap_dues_rows(
    spec: PageSpec,
    rng: random.Random,
    boxes: list[LineBox],
    rules: list[Rule],
    *,
    y: int,
    bx0: int,
    by1: int,
    vx: int,
    val_w: int,
    indent: int,
    step: int,
) -> tuple[int, int]:
    """One section's ``label ······ value`` rows (a due joined to its figure by a dotted leader);
    return (the y reached, the section's running subtotal)."""
    subtotal = 0
    for _ in range(rng.randint(4, 9)):  # many dues per holding, like the real length-rolls
        if y + spec.line_height > by1:
            break
        label, value = rng.choice(_LANDSKAP_DUES), rng.randint(1, 30)
        subtotal += value
        lw = min(
            vx - (bx0 + indent) - spec.line_height,
            max(spec.line_height, int(len(label) * spec.px_per_char)),
        )
        boxes.append(_box(label, bx0 + indent, y, lw, spec.line_height, spec, rng, "reg_label"))
        boxes.append(
            _box(
                str(value),
                vx,
                y,
                val_w,
                spec.line_height,
                spec,
                rng,
                "reg_value",
                "right",
                printed=spec.typeset_furniture,
            )
        )  # figures typeset — the model smudges lone digits
        ly = y + int(spec.line_height * spec.baseline_frac)  # dotted leader on the baseline
        rules.append(Rule(x1=bx0 + indent + lw + 8, y1=ly, x2=vx - 8, y2=ly, width=1, style="dashed"))
        y += step
    return y, subtotal


def _landskap_summa_row(
    spec: PageSpec,
    rng: random.Random,
    subtotal: int,
    boxes: list[LineBox],
    rules: list[Rule],
    *,
    y: int,
    bx0: int,
    vx: int,
    val_w: int,
    indent: int,
    step: int,
) -> None:
    """A closing Summa row: a rule over the values, the label, and the section's right-aligned subtotal."""
    rules.append(Rule(x1=bx0 + indent, y1=y - step // 3, x2=vx + val_w, y2=y - step // 3, width=1))
    boxes.append(_box("Summa", bx0 + indent, y, indent * 3, spec.line_height, spec, rng, "reg_header"))
    boxes.append(
        _box(
            str(subtotal),
            vx,
            y,
            val_w,
            spec.line_height,
            spec,
            rng,
            "reg_value",
            "right",
            printed=spec.typeset_furniture,
        )
    )


def _landskap(
    spec: PageSpec, paragraphs: list[str], rng: random.Random, measure: Measure | None = None
) -> Layout:
    """A boskapslängd / landskapshandling — a sparse 16th-17th c. tax & cattle roll: DISPLAY section headers
    (a parish or tenant name, written larger) over indented ``label ······ value`` rows (a due joined to its
    small figure by a dotted leader), grouped into sections with generous whitespace and a wide right margin.

    This is the sparse, display-headed register the corpus's oldest volumes are full of — which ``register``
    approximated structurally but never rendered in this form. Values are single figures fitted to a narrow
    column and labels are single words, so nothing overflows its slot. Content is generated from a tax-roll
    vocabulary; ``paragraphs`` are ignored (the form's text is intrinsic, like ``semi_table``).
    """
    extra, (bx0, by0, bx1, by1) = _page_elements(spec, rng)
    col_w = int((bx1 - bx0) * 0.62)  # entries fill the left ~62%; real rolls leave a wide right margin
    val_w = max(int(1.6 * spec.line_height), int(0.09 * (bx1 - bx0)))  # narrow value column
    vx = bx0 + col_w - val_w
    indent = int(spec.line_height * 1.3)  # items are indented under their section header
    step = int(spec.line_height * (1.0 + max(spec.line_gap, 0.3)))  # real rolls pack the entries fairly tight
    disp_h = int(spec.line_height * 1.4)  # DISPLAY-scale section header (enlarged heading)
    boxes: list[LineBox] = list(extra)
    rules: list[Rule] = []
    y = by0
    while y + disp_h + 2 * step <= by1:  # only start a section that can show a header + a couple of rows
        head = f"{rng.choice(_LANDSKAP_PLACES)} {rng.choice(_LANDSKAP_KINDS)}"
        hw = min(col_w, max(spec.line_height, int(len(head) * spec.px_per_char * 1.05)))
        boxes.append(_box(head, bx0, y, hw, disp_h, spec, rng, "reg_header"))
        y += int(disp_h * 1.05)
        y, subtotal = _landskap_dues_rows(
            spec, rng, boxes, rules, y=y, bx0=bx0, by1=by1, vx=vx, val_w=val_w, indent=indent, step=step
        )
        if y + spec.line_height <= by1 and rng.random() < 0.7:  # a Summa figure closes most sections
            _landskap_summa_row(
                spec, rng, subtotal, boxes, rules, y=y, bx0=bx0, vx=vx, val_w=val_w, indent=indent, step=step
            )
            y += step
        y += int(step * 0.45)  # a modest blank gap between sections (they sit close on the real rolls)
    return Layout(spec=spec, boxes=boxes, rules=rules)


TEMPLATES = {
    "single_column": _single_column,
    "multi_column": _multi_column,
    "semi_table": _semi_table,
    "columns": _columns,
    "key_value": _key_value,
    "spread": _spread,
    "letter": _letter,
    "register": _register,
    "ledger": _ledger,
    "landskap": _landskap,
    "blank_leaf": _blank_leaf,
}
_TEMPLATES = TEMPLATES  # backwards-compatible alias for pre-rename internal refs


def build_layout(
    spec: PageSpec, paragraphs: list[str], *, seed: int = 0, measure: Measure | None = None
) -> Layout:
    """Build a page layout from a template + text. Deterministic given ``seed``.

    Pass ``measure`` (a renderer's ``measure_width``) so lines are wrapped to the renderer's actual
    width — keeping font size uniform within a paragraph. Omit it to wrap by a px_per_char estimate.
    """
    fn = TEMPLATES.get(spec.template)
    if fn is None:
        raise ValueError(f"unknown template {spec.template!r}; options: {sorted(TEMPLATES)}")
    rng = random.Random(seed)
    layout = fn(spec, paragraphs, rng, measure)
    if spec.drop_cap:
        layout = _apply_drop_cap(layout, rng)
    if spec.insertion_prob > 0.0 or spec.deletion_prob > 0.0 or spec.underline_prob > 0.0:
        layout = _apply_edits(layout, rng)
    return layout


def to_single_leaf(layout: Layout, prefer: LeafPrefer = "denser") -> Layout:
    """Reduce a two-page-spread replay to ONE leaf (we only train on single pages): if the boxes fall into a
    left and a right half separated by a central gutter, keep one half and re-origin the canvas onto it.

    ``prefer`` = 'denser' (the fuller leaf, default), 'blank' (the emptier leaf — a near-blank training
    sample), 'left', or 'right'. A layout that is already a single leaf is returned unchanged.
    """
    boxes = layout.boxes
    if not boxes:
        return layout
    cx = layout.spec.width / 2
    left = [b for b in boxes if b.x + b.w / 2 < cx]
    right = [b for b in boxes if b.x + b.w / 2 >= cx]
    if not left or not right or min(len(left), len(right)) < 0.2 * len(boxes):
        return layout  # a single leaf already — nothing to split
    if prefer == "left":
        keep = left
    elif prefer == "right":
        keep = right
    elif prefer == "blank":
        keep = left if len(left) <= len(right) else right  # the emptier leaf
    else:
        keep = right if len(right) >= len(left) else left  # denser
    x0 = min(b.x for b in keep)
    x1 = max(b.x + b.w for b in keep)
    pad = int(0.03 * (x1 - x0))
    shift = x0 - pad
    new_boxes = [b.model_copy(update={"x": b.x - shift}) for b in sorted(keep, key=lambda b: (b.y, b.x))]
    # Carry only the kept leaf's furniture rules (by rule-midpoint x), re-origined onto the new canvas —
    # otherwise a spread replay keeps the other leaf's rules, unshifted, drawn across this page.
    new_rules = [
        r.model_copy(update={"x1": r.x1 - shift, "x2": r.x2 - shift})
        for r in layout.rules
        if x0 <= (r.x1 + r.x2) / 2 <= x1
    ]
    return Layout(
        spec=layout.spec.model_copy(update={"width": (x1 - x0) + 2 * pad}),
        boxes=new_boxes,
        rules=new_rules,
    )
