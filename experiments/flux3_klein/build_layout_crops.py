"""FLUX.3 Klein LoRA training crops with layout captions, from the HTR ground truth (PAGE XML pages).

Archival spreads are ~5000 px wide with 60-120 lines, so a whole page at FLUX's ~1 MP shrinks the letters
to a few pixels. Each crop instead covers consecutive WHOLE lines of one text column (a text region plus
the margin notes, headings and page numbers beside, above or below it), scaled so the line spacing is
about ``--line-spacing`` px and saved at the exact size of a NeMo bucket. Its caption follows BFL's
layout-prompt guide: the collection, the volume's years and the scan's look (the style handles), every
line cited as ``<Sv_Text_i>`` with special lines placed in words, then FLUX.3's layout JSON
(``[y_min, x_min, y_max, x_max]`` on 0-1000, y first) quoting each line's transcription.

Input: the Transkribus-export layout of the GT folders, ``<root>/<collection>/<volume>/<volume>/`` with the
scans and a ``page/`` folder of PAGE XML named like them. Output: an imagefolder dataset per split
(``<out>/<split>/<id>.jpg`` + ``metadata.jsonl``, one ``CropRecord`` per crop), loadable with
``datasets.load_dataset("<out>/<split>")``.

    uv run --no-sync python experiments/flux3_klein/build_layout_crops.py \
        --gt-root ~/Desktop/data/GT-Lejonet_HTR_202602 --max-pages 40 --out-dir data_flux3/layout_crops_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

import numpy as np
from PIL import Image
from pydantic import BaseModel

from diffu.data.parse import iter_local, local_name, page_line_text, parse_points

Image.MAX_IMAGE_PIXELS = None  # archival spreads exceed PIL's decompression-bomb default

type Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels

SPLITS = ("train", "val")
SCAN_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
YEARS = re.compile(r"(1[5-9]\d\d)(?:\s*[-–]\s*(1[5-9]\d\d))?(-tal|-luku)?")

# NeMo's image buckets for --resolution_preset 1024p (MultiTierBucketCalculator, quantization 64), (w, h).
# Preprocessing fits an image to the bucket nearest its aspect by a cover-resize and a center crop, which
# would trim a crop's edge lines out from under their boxes; a crop saved at a bucket's exact size is not cut.
BUCKETS: list[tuple[int, int]] = [
    (768, 1344), (832, 1216), (896, 1152), (1024, 1024), (1152, 896),
    (1216, 832), (1280, 768), (1344, 768), (1408, 704), (1472, 640),
]  # fmt: skip
WIDEST_BUCKET = max(width for width, _ in BUCKETS)

# A line's box is its slot on the page, in line pitches: PAGE polygons include ascenders and flourishes, so
# their own boxes overlap the neighbouring lines'.
SLOT_ABOVE_BASELINE = 0.7
SLOT_BELOW_BASELINE = 0.3
SLOT_SIDE_MARGIN = 0.15
WHOLE_LINE = 0.9  # share of a line's width inside a crop that counts as the whole line
CUT_LINE = 0.15  # above this (and below WHOLE_LINE) a line is cut at the side and the crop is dropped

# Mean HSV saturation (0-1) below which a crop is a greyscale (microfilm) scan: those measure 0.0, colour
# scans 0.08-0.39.
GREY_SATURATION = 8.0 / 255
# Colour names by level, from the GT collections: colour paper is hue 28-48 degrees, saturation 0.09-0.35;
# ink is brown (hue 22-45) at value 0.23-0.49; greyscale paper is value 0.71-0.91 and its ink 0.05-0.45.
NEUTRAL_PAPER = 0.06  # paper saturation below which it is named by brightness
NEUTRAL_PAPERS = ((0.78, "grey"), (0.88, "light grey"), (math.inf, "white"))  # by value
TINTED_PAPERS = ((0.12, "off-white"), (0.22, "cream"), (math.inf, "yellowed"))  # by saturation
BROWN_INK_SATURATION = 0.15
BROWN_INK_HUES = (15.0, 55.0)  # degrees
BROWN_INKS = ((0.30, "dark brown"), (0.42, "brown"), (math.inf, "faded brown"))  # by value
GREY_INKS = ((0.15, "black"), (0.35, "dark grey"), (math.inf, "faded grey"))  # by value

NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")

# The GT collection folder (minus "GT-" and the export date) -> the name its captions use: the style handle
# that picks a collection's hand, paper and ink at generation time.
COLLECTION_TITLES = {
    "FNA_1500": "the 16th-century records of the National Archives of Finland",
    "FNA_finnish_court_records": "Finnish court records in the National Archives of Finland",
    "FNA_finnish_letter_recipes": "a Finnish book of letters and recipes",
    "JLF_jamtlands_domsaga": "the court records of Jämtlands domsaga",
    "RA_1634_ars_regeringsform": "the 1634 Instrument of Government (1634 års regeringsform)",
    "RA_alvsborgs_losen": "the Älvsborgs lösen tax rolls",
    "RA_arjeplogs_tingslag": "the court minutes of Arjeplogs tingslag",
    "RA_bergmastare_i_nora": "the hammer-court minutes of Bergmästaren i Nora",
    "RA_bergskollegium_rattsprotokoll": "the court minutes of Bergskollegium",
    "RA_bergskollegium_relationer_och_skrivelser": "the reports and letters of Bergskollegium",
    "RA_bergskollegium_stora_kopparbergs_bergslag": "the Bergskollegium records of Stora Kopparbergs bergslag",
    "RA_boskapslangder": "the boskapslängder livestock tax rolls",
    "RA_carl_fredrik_pahlmans_resejournaler": "Carl Fredrik Påhlman's travel journals",
    "RA_frihetstidens_utskottshandlingar": "the committee papers of the Age of Liberty diets",
    "RA_gota_hovratt_brottsmalsprotokoll": "the criminal-case minutes of Göta hovrätt",
    "RA_goteborgs_poliskammare": "the records of Göteborgs poliskammare",
    "RA_harnosands_domkapitlet": "the records of Härnösands domkapitel",
    "RA_jonkopings_radhusratts_och_magistrat": "the records of Jönköpings rådhusrätt och magistrat",
    "RA_krigsarkivet_dombocker": "the regimental court records of Krigsarkivet",
    "RA_landskapshandlingar": "the landskapshandlingar provincial accounts",
    "RA_landtmatarnes_ankomne_ordres": "the land surveyors' received orders (Landtmätarnes ankomne ordres)",
    "RA_riksregistraturet": "Riksregistraturet, the royal chancery register",
    "RA_strangnas_domkapitel": "the records of Strängnäs domkapitel",
    "RA_strodda_kamerala_handlingar": "the strödda kamerala handlingar fiscal records",
    "RA_svea_hovratt_koncept": "the draft minutes of Svea hovrätt",
    "RA_svea_hovratt_renskrivna_protokoll": "the fair-copy minutes of Svea hovrätt",
    "RA_trolldomskommissionen": "the records of the witchcraft commissions (trolldomskommissionen)",
    "RA_uppvidinge_haradsratt": "the court records of Uppvidinge häradsrätt",
    "RA_vase_haradsratt": "the court records of Väse häradsrätt",
    "RA-X_beskrivning_over_provinsen_gastrikland": "a description of the province of Gästrikland",
    "RA-X_gbg_1790-1794": "a Göteborg register",
    "RA_yngre_geometriska_kartor": "the younger geometric maps (yngre geometriska kartor)",
    "SSA_allmanna_barnhuset": "the records of Allmänna barnhuset in Stockholm",
    "SSA_borgmastare_och_rad_fore_1636": "the minutes of Stockholm's borgmästare och råd",
    "SSA_botkyrka_kyrkoarkiv": "the parish records of Botkyrka",
    "SSA_brannkyrka_kyrkoarkiv": "the parish records of Brännkyrka",
    "SSA_byggningskollegium": "the records of Stockholm's byggningskollegium",
    "SSA_deputerade": "the records of Stockholm's deputerade",
    "SSA_fritjof_englund": "Fritjof Englund's papers",
    "SSA_grödinge_kyrkoarkiv": "the parish records of Grödinge",
    "SSA_hall-_och_manufakturratten": "the records of Hall- och manufakturrätten",
    "SSA_handelskollegiet": "the records of Stockholm's handelskollegium",
    "SSA_justitiekollegiet_och_formyndarkammaren": "the records of Justitiekollegiet och förmyndarkammaren",
    "SSA_kungliga_hovkonsistoriet": "the records of Kungliga hovkonsistoriet",
    "SSA_norra_forstadens_kamnarsratt": "the court records of Norra förstadens kämnärsrätt",
    "SSA_norra_forstadens_vastra_kamnarsratt": "the court records of Norra förstadens västra kämnärsrätt",
    "SSA_overstathallarambetets_aldre_kansli": "the records of Överståthållarämbetet",
    "SSA_politikollegiet": "the records of Stockholm's politikollegium",
    "SSA_sodra_forstadens_kamnarsratt": "the court records of Södra förstadens kämnärsrätt",
    "SSA_sollentuna_haradsratt": "the court records of Sollentuna häradsrätt",
    "SSA_sotholms_haradsratt": "the court records of Sotholms häradsrätt",
    "SSA_spånga_kyrkoarkiv": "the parish records of Spånga",
    "SSA_stadens_kamnarsratt": "the court records of Stockholm's stadens kämnärsrätt",
    "SSA_stockholm_magistrat_och_radhusratt": "the records of Stockholms magistrat och rådhusrätt",
    "SSA_stockholms_domkapitel": "the records of Stockholms domkapitel",
    "SSA_svartlosa_haradsratt": "the court records of Svartlösa häradsrätt",
    "VKaariainen_1600-luku_GT": "17th-century Finnish records",
}
# PAGE region types (``type`` or Transkribus' ``structure {type:...}``) a line's desc names; other lines are
# plain "line of text".
ROLE_NAMES = {
    "marginalia": "marginal note",
    "page-number": "page number",
    "heading": "heading",
    "header": "running header",
    "catch-word": "catchword",
    "signature-mark": "signature mark",
    "footer": "footer",
}


class CropSettings(BaseModel):
    """How pages are cut into crops (pixels are after scaling unless noted)."""

    line_spacing: float = 64.0  # target line pitch
    min_line_spacing: float = 40.0  # crops scaled below this pitch are skipped
    min_merge_spacing: float = 48.0  # smallest pitch a margin note may shrink its column to
    min_width: int = 640  # narrower regions are only cropped together with a wide one
    min_lines: int = 3
    overlap_lines: int = 2  # lines shared by consecutive windows of a column


class Line(BaseModel):
    polygon: list[tuple[float, float]]
    baseline_y: float
    text: str
    role: str = ""  # its region's PAGE type when ROLE_NAMES names it

    @property
    def box(self) -> Box:
        xs, ys = zip(*self.polygon, strict=True)
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def middle_x(self) -> float:
        return (self.box[0] + self.box[2]) / 2


class Region(BaseModel):
    lines: list[Line]

    @property
    def box(self) -> Box:
        boxes = [line.box for line in self.lines]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


class Column(BaseModel):
    """A wide text region and the narrow regions (margin notes, headings, page numbers) that joined it."""

    anchor: Box  # the wide region's own box
    left: float
    right: float
    lines: list[Line]

    @classmethod
    def of(cls, region: Region) -> Column:
        box = region.box
        return cls(anchor=box, left=box[0], right=box[2], lines=list(region.lines))

    def gap_to(self, box: Box) -> float:
        """Vertical distance from the anchor region to ``box``; 0 when they overlap."""
        return max(0.0, max(self.anchor[1], box[1]) - min(self.anchor[3], box[3]))

    def width_with(self, box: Box) -> float:
        return max(self.right, box[2]) - min(self.left, box[0])

    def add(self, region: Region) -> None:
        self.lines.extend(region.lines)
        self.left = min(self.left, region.box[0])
        self.right = max(self.right, region.box[2])


class PageGeometry(BaseModel):
    """What cropping a column needs to know about its page: size (px), line pitch and every line."""

    size: tuple[int, int]
    pitch: float
    lines: list[Line]


class ColumnFrame(BaseModel):
    """A column's padded x-range (page px) and the bucket its crops are saved at."""

    left: float
    right: float
    bucket: tuple[int, int]

    @property
    def scale(self) -> float:
        return self.bucket[0] / (self.right - self.left)

    @property
    def span(self) -> float:
        """The crop height in page pixels."""
        return self.bucket[1] / self.scale


class Crop(BaseModel):
    """A crop of a page: its rectangle (page px), bucket size, whole lines and the lines its top or bottom cuts."""

    rect: Box
    size: tuple[int, int]
    pitch: float
    lines: list[Line]
    fragments: list[Line]

    def layout_box(self, box: Box) -> list[int]:
        """A page rectangle as a FLUX.3 layout box: ``[y_min, x_min, y_max, x_max]`` on 0-1000, clamped."""
        x0, y0, x1, y1 = self.rect
        width, height = x1 - x0, y1 - y0
        relative = (
            (box[1] - y0) / height,
            (box[0] - x0) / width,
            (box[3] - y0) / height,
            (box[2] - x0) / width,
        )
        return [min(1000, max(0, round(1000 * v))) for v in relative]

    def slot(self, line: Line) -> list[int]:
        """A line's layout box: its x-extent plus a small margin, one line pitch around its baseline."""
        margin = SLOT_SIDE_MARGIN * self.pitch
        top = line.baseline_y - SLOT_ABOVE_BASELINE * self.pitch
        bottom = line.baseline_y + SLOT_BELOW_BASELINE * self.pitch
        return self.layout_box((line.box[0] - margin, top, line.box[2] + margin, bottom))

    def visible_box(self, line: Line) -> list[int]:
        """The layout box of the part of a cut-off line inside the crop."""
        return self.layout_box(
            (line.box[0], max(line.box[1], self.rect[1]), line.box[2], min(line.box[3], self.rect[3]))
        )

    def place(self, x: float, y: float) -> str:
        """A page point's position on the crop in words: "in the upper right corner", "at the top center"."""
        x0, y0, x1, y1 = self.rect
        return _place((x - x0) / (x1 - x0), (y - y0) / (y1 - y0))

    def cut_edge(self, line: Line) -> str:
        return "top" if line.baseline_y < self.rect[1] else "bottom"


class ScanLook(BaseModel):
    """What a crop's scan looks like, in plain words for its caption."""

    model_config = {"frozen": True}

    greyscale: bool
    paper: str  # e.g. "yellowed", "light grey"
    ink: str  # e.g. "dark brown", "black"


class Provenance(BaseModel):
    """Where a crop comes from, as its caption names it."""

    model_config = {"frozen": True}

    title: str  # the collection
    years: str  # the volume's, '' when its name has none


class LayoutRow(BaseModel):
    """One element of a FLUX.3 layout prompt."""

    id: str
    bbox: list[int]
    desc: str


class LineRecord(BaseModel):
    """A crop's text line for evaluation: its layout row id, transcription, PAGE role and box."""

    id: str
    text: str
    role: str
    bbox: list[int]


class Caption(BaseModel):
    text: str  # the prompt: the caption paragraph, a space and the layout rows as JSON
    layout: list[LayoutRow]
    lines: list[LineRecord]


class CropRecord(BaseModel):
    """A ``metadata.jsonl`` row: the imagefolder fields NeMo reads (``file_name``, ``caption``) and the rest."""

    file_name: str
    caption: str
    collection: str
    page: str
    width: int
    height: int
    greyscale: bool
    paper: str
    ink: str
    layout: list[LayoutRow]
    lines: list[LineRecord]

    @property
    def provenance(self) -> Provenance:
        return provenance_of(self.collection, Path(self.page))

    @property
    def look(self) -> ScanLook:
        return ScanLook(greyscale=self.greyscale, paper=self.paper, ink=self.ink)


class Page(BaseModel):
    """A ground-truth page with a scan and transcribed text."""

    collection: str
    path: Path  # the PAGE XML, relative to the GT root
    scan: Path
    regions: list[Region]

    @property
    def id(self) -> str:
        return hashlib.sha1(str(self.path).encode()).hexdigest()[:12]

    @property
    def provenance(self) -> Provenance:
        return provenance_of(self.collection, self.path)


def read_records(metadata: Path) -> list[CropRecord]:
    """The crop records of a split's ``metadata.jsonl``."""
    with metadata.open(encoding="utf-8") as rows:
        return [CropRecord.model_validate_json(row) for row in rows]


def provenance_of(collection: str, page_xml: Path) -> Provenance:
    """The archive and years a caption names for a page: ``<collection>/<volume>/<volume>/page/<file>.xml``."""
    return Provenance(title=collection_title(collection), years=volume_years(page_xml.parent.parent.name))


def collection_title(collection: str) -> str:
    """The caption name of a GT collection folder (its slug for a collection COLLECTION_TITLES lacks)."""
    key = re.sub(r"^GT-|_\d{8}$", "", collection)
    return COLLECTION_TITLES.get(key, key.replace("_", " "))


def volume_years(volume: str) -> str:
    """The years in a volume folder's name ('1676–1677', '1777', '1690s'), preferring a range; '' if none."""
    found = list(YEARS.finditer(volume))
    if not found:
        return ""
    ranges = [match for match in found if match.group(2)]
    first, last, century_suffix = (ranges or found)[0].groups()
    if last:
        return f"{first}–{last}"
    if century_suffix:
        return f"{first}s"
    return first


def scan_look(img: Image.Image) -> ScanLook:
    """Greyscale or colour, and names for the paper's colour (the median of the lightest 40% of pixels) and
    the ink's (the median of the darkest 2%)."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32).reshape(-1, 3) / 255
    value = hsv[:, 2]
    greyscale = float(hsv[:, 1].mean()) < GREY_SATURATION
    _, paper_saturation, paper_value = np.median(hsv[value >= np.percentile(value, 60)], axis=0)
    ink_hue, ink_saturation, ink_value = np.median(hsv[value <= np.percentile(value, 2)], axis=0)
    if greyscale:  # a microfilm's leftover saturation is scanner noise
        paper_saturation = ink_saturation = 0.0
    return ScanLook(
        greyscale=greyscale,
        paper=_paper_colour(paper_saturation, paper_value),
        ink=_ink_colour(ink_hue * 360, ink_saturation, ink_value),
    )


def _paper_colour(saturation: float, value: float) -> str:
    if saturation < NEUTRAL_PAPER:
        return _band(value, NEUTRAL_PAPERS)
    return _band(saturation, TINTED_PAPERS)


def _ink_colour(hue: float, saturation: float, value: float) -> str:
    low, high = BROWN_INK_HUES
    if saturation < BROWN_INK_SATURATION or not low <= hue <= high:
        return _band(value, GREY_INKS)
    return _band(value, BROWN_INKS)


def _band(level: float, bands: tuple[tuple[float, str], ...]) -> str:
    """The name of the first band ``level`` is below; the last band's limit is infinite."""
    return next(name for limit, name in bands if level < limit)


def parse_page(xml_path: Path) -> list[Region]:
    """A PAGE XML page's text regions in reading order, each with its transcribed lines from top to bottom."""
    page = iter_local(ET.parse(xml_path).getroot(), "Page")[0]
    ordered: list[tuple[int, Region]] = []
    for position, element in enumerate(iter_local(page, "TextRegion")):
        role = _region_role(element)
        parsed = (_parse_line(child, role) for child in element if local_name(child.tag) == "TextLine")
        lines = sorted((line for line in parsed if line is not None), key=lambda line: line.baseline_y)
        if lines:
            ordered.append((_reading_order(element, position), Region(lines=lines)))
    return [region for _, region in sorted(ordered, key=lambda item: item[0])]


def _region_role(region: ET.Element) -> str:
    structure = re.search(r"structure \{type:([^;}]+)", region.attrib.get("custom", ""))
    role = region.attrib.get("type") or (structure.group(1) if structure else "")
    return role if role in ROLE_NAMES else ""


def _reading_order(region: ET.Element, position: int) -> int:
    """The region's Transkribus reading-order index, else its position in the file."""
    match = re.search(r"readingOrder \{index:(\d+);", region.attrib.get("custom", ""))
    return int(match.group(1)) if match else position


def _parse_line(element: ET.Element, role: str) -> Line | None:
    """A TextLine with its transcription, outline and baseline height; None without text or outline."""
    text = (page_line_text(element) or "").strip()
    coords = _child(element, "Coords")
    if not text or coords is None:
        return None
    polygon = parse_points(coords.attrib.get("points", ""))
    if len(polygon) < 3:
        return None
    baseline = _child(element, "Baseline")
    points = parse_points(baseline.attrib.get("points", "")) if baseline is not None else []
    baseline_y = statistics.mean(y for _, y in points) if points else max(y for _, y in polygon)
    return Line(polygon=polygon, baseline_y=baseline_y, text=text, role=role)


def _child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if local_name(child.tag) == name), None)


def line_spacing(regions: list[Region]) -> float | None:
    """Median vertical distance between consecutive baselines within regions (the page's line pitch)."""
    gaps = [
        b.baseline_y - a.baseline_y
        for region in regions
        for a, b in zip(region.lines, region.lines[1:], strict=False)
        if b.baseline_y > a.baseline_y
    ]
    return statistics.median(gaps) if gaps else None


def _x_overlap(box: Box, rect: Box) -> float:
    """Fraction of ``box``'s width inside ``rect``'s x-range."""
    inside = max(0.0, min(box[2], rect[2]) - max(box[0], rect[0]))
    return inside / max(box[2] - box[0], 1e-6)


def text_columns(regions: list[Region], pitch: float, settings: CropSettings) -> list[list[Line]]:
    """The lines of each wide text region together with the narrow regions that belong beside it.

    A region too narrow to be cropped on its own (a margin note, heading, page number) joins the nearest
    wide region it overlaps vertically or sits within two line pitches of, if the merged column, scaled to
    fit the widest bucket, keeps a line pitch of at least ``min_merge_spacing``; it is then cropped with that
    region's lines, so the crops show page structure and not just body text. Lines come back sorted by
    baseline.
    """
    natural_scale = min(1.0, settings.line_spacing / pitch)
    is_wide = [(r.box[2] - r.box[0] + pitch) * natural_scale >= settings.min_width for r in regions]
    columns = [Column.of(region) for region, wide in zip(regions, is_wide, strict=True) if wide]
    for region, wide in zip(regions, is_wide, strict=True):
        if wide:
            continue
        box = region.box
        homes = [c for c in columns if c.gap_to(box) <= 2 * pitch and _merge_fits(c, box, pitch, settings)]
        if homes:
            min(homes, key=lambda c: c.gap_to(box)).add(region)
    return [sorted(c.lines, key=lambda line: line.baseline_y) for c in columns]


def _merge_fits(column: Column, box: Box, pitch: float, settings: CropSettings) -> bool:
    """Whether ``column`` plus ``box``, scaled to fit the widest bucket, keeps a legible line pitch."""
    span = column.width_with(box) + pitch
    scale = min(1.0, settings.line_spacing / pitch, WIDEST_BUCKET / span)
    return pitch * scale >= settings.min_merge_spacing


def crops_for_page(regions: list[Region], size: tuple[int, int], settings: CropSettings) -> list[Crop]:
    """Crops over consecutive whole lines of each text column (see ``text_columns``).

    A column's crops all take the size of one NeMo bucket (the one nearest the column's width at the target
    line pitch whose height still fits on the page). Windows of lines overlap by ``overlap_lines``. A crop
    holds every line whose baseline lies inside it and whose width is (almost) whole; a crop that cuts a line
    at the side (unlabelled visible text) is dropped. Lines cut by the top or bottom edge are its
    ``fragments``.
    """
    pitch = line_spacing(regions)
    if pitch is None:
        return []
    page = PageGeometry(size=size, pitch=pitch, lines=[line for region in regions for line in region.lines])
    return [
        crop
        for lines in text_columns(regions, pitch, settings)
        for crop in _column_crops(lines, page, settings)
    ]


def _column_crops(lines: list[Line], page: PageGeometry, settings: CropSettings) -> list[Crop]:
    frame = _column_frame(lines, page, settings)
    if frame is None:
        return []
    per_window = _lines_per_window(lines, frame.span, page.pitch)
    if per_window < settings.min_lines:
        return []
    starts = _window_starts(len(lines), per_window, settings.overlap_lines)
    crops = (_crop(_window_rect(lines[i : i + per_window], frame, page), frame.bucket, page) for i in starts)
    return [crop for crop in crops if crop is not None and len(crop.lines) >= settings.min_lines]


def _column_frame(lines: list[Line], page: PageGeometry, settings: CropSettings) -> ColumnFrame | None:
    """The column's padded x-range and the bucket nearest its width at the target pitch; None if unfit."""
    pad = 0.5 * page.pitch
    left = max(0.0, min(line.box[0] for line in lines) - pad)
    right = min(float(page.size[0]), max(line.box[2] for line in lines) + pad)
    natural_width = (right - left) * min(1.0, settings.line_spacing / page.pitch)
    fitting = [b for b in BUCKETS if b[1] * (right - left) / b[0] <= page.size[1]]
    if natural_width < settings.min_width or not fitting:
        return None

    def width_error(bucket: tuple[int, int]) -> float:
        return abs(bucket[0] - natural_width)

    frame = ColumnFrame(left=left, right=right, bucket=min(fitting, key=width_error))
    if page.pitch * frame.scale < settings.min_line_spacing:
        return None
    return frame


def _lines_per_window(lines: list[Line], span: float, pitch: float) -> int:
    count = len(lines)
    while count > 1 and lines[count - 1].baseline_y - lines[0].baseline_y + 2 * pitch > span:
        count -= 1
    return count


def _window_starts(total: int, size: int, overlap: int) -> list[int]:
    """First-line indices of windows of ``size`` lines sharing ``overlap``; the last ends at the column's end."""
    starts = list(range(0, max(total - size, 0) + 1, max(1, size - overlap)))
    if starts[-1] != total - size:
        starts.append(total - size)
    return starts


def _window_rect(window: list[Line], frame: ColumnFrame, page: PageGeometry) -> Box:
    middle = (window[0].baseline_y + window[-1].baseline_y) / 2 - 0.25 * page.pitch
    top = min(max(0.0, middle - frame.span / 2), page.size[1] - frame.span)
    return (frame.left, top, frame.right, top + frame.span)


def _crop(rect: Box, bucket: tuple[int, int], page: PageGeometry) -> Crop | None:
    """The crop at ``rect``; None when it cuts a line at the side."""
    inside = [line for line in page.lines if rect[1] <= line.baseline_y <= rect[3]]
    if any(CUT_LINE < _x_overlap(line.box, rect) < WHOLE_LINE for line in inside):
        return None
    return Crop(
        rect=rect,
        size=bucket,
        pitch=page.pitch,
        lines=[line for line in inside if _x_overlap(line.box, rect) >= WHOLE_LINE],
        fragments=[line for line in page.lines if _is_cut_off(line, rect, page.pitch)],
    )


def _is_cut_off(line: Line, rect: Box, pitch: float) -> bool:
    """Whether the crop's top or bottom edge cuts the line: baseline outside, a visible part of its ink inside."""
    if rect[1] <= line.baseline_y <= rect[3] or _x_overlap(line.box, rect) < 0.5:
        return False
    return min(line.box[3], rect[3]) - max(line.box[1], rect[1]) > 0.25 * pitch


def caption_for(crop: Crop, provenance: Provenance, look: ScanLook) -> Caption:
    """The FLUX.3 layout prompt of a crop: one English paragraph citing every element as ``<id>``, a space, the
    layout rows as JSON.

    Follows the annotation rules FLUX.3 was trained on (the ``t2i_bbox`` system prompt of its prompt upsampler)
    and BFL's layout-prompt guide: medium, setting and light first; each ``<id>`` cited beside the words that
    name it, with its position in words as well as in its box; visible details (paper and ink colour) rather
    than names alone; every visible text a row quoting its words with style and alignment. A text line's id is
    its language code + ``_Text_`` + its index in reading order (``Sv_Text_1``). Lines cut by the top or bottom
    edge get a ``fragment_k`` row, so the model learns that unprompted ink appears only where a row says so.
    """
    numbered = [
        (f"Sv_Text_{i}", line)
        for i, line in enumerate(sorted(crop.lines, key=lambda line: (line.baseline_y, line.box[0])), start=1)
    ]
    fragments = [
        (f"fragment_{k}", line)
        for k, line in enumerate(sorted(crop.fragments, key=lambda line: line.baseline_y), 1)
    ]
    column = _body_column([line for _, line in numbered])
    layout = [
        LayoutRow(id="paper_1", bbox=[0, 0, 1000, 1000], desc=_paper_desc(look)),
        *(
            LayoutRow(
                id=row_id,
                bbox=crop.slot(line),
                desc=_line_desc(_line_kind(line, column, crop.pitch), line, look),
            )
            for row_id, line in numbered
        ),
        *(
            LayoutRow(id=row_id, bbox=crop.visible_box(line), desc=_fragment_desc(crop.cut_edge(line), look))
            for row_id, line in fragments
        ),
    ]
    sentences = [
        _opening_sentence(provenance, look, numbered),
        *_role_sentences(numbered, column, crop),
        *_fragment_sentences(fragments, crop),
    ]
    rows = json.dumps([row.model_dump() for row in layout], ensure_ascii=False)
    return Caption(
        text=f"{' '.join(sentences)} {rows}",
        layout=layout,
        lines=[
            LineRecord(id=row_id, text=line.text, role=line.role, bbox=crop.slot(line))
            for row_id, line in numbered
        ],
    )


def _body_column(lines: list[Line]) -> tuple[float, float]:
    """The x-range of the crop's main text (of all its lines when every line has a role)."""
    body = [line for line in lines if not line.role] or lines
    return min(line.box[0] for line in body), max(line.box[2] for line in body)


def _line_kind(line: Line, column: tuple[float, float], pitch: float) -> str:
    """How a line's desc opens: its role, or how a main-text line sits in its column."""
    if line.role:
        return ROLE_NAMES[line.role].capitalize()
    indent = (line.box[0] - column[0]) / pitch
    gap = (column[1] - line.box[2]) / pitch
    if indent > 2 and gap > 2 and abs(indent - gap) < 1.5:
        return "Centered handwritten line of text"
    if indent > 1.5:
        return "Indented handwritten line of text"
    return "Handwritten line of text"


def _line_desc(kind: str, line: Line, look: ScanLook) -> str:
    return f'{kind} reading "{line.text}", cursive script in {look.ink} ink.'


def _paper_desc(look: ScanLook) -> str:
    scan = ", in a greyscale scan." if look.greyscale else "."
    return f"Aged {look.paper} paper of a bound archival volume{scan}"


def _fragment_desc(edge: str, look: ScanLook) -> str:
    part = "lower" if edge == "top" else "upper"
    return (
        f"The {part} part of a handwritten line in {look.ink} ink, cut off by the {edge} edge of the scan; "
        "its words are not legible."
    )


def _opening_sentence(provenance: Provenance, look: ScanLook, numbered: list[tuple[str, Line]]) -> str:
    roles = Counter(line.role for _, line in numbered if line.role)
    parts = [
        f"a {ROLE_NAMES[r]}" if n == 1 else f"{_count(n)} {ROLE_NAMES[r]} lines" for r, n in roles.items()
    ]
    including = f", including {' and '.join(parts)}" if parts else ""
    scan = "greyscale archival scan" if look.greyscale else "archival scan"
    when = f", dated {provenance.years}" if provenance.years else ""
    opening = (
        f"Flat, evenly lit {scan} of a handwritten page from {provenance.title}{when}. The {look.paper} paper "
        f"<paper_1> carries {len(numbered)} lines of cursive handwriting in {look.ink} ink{including}"
    )
    body = [row_id for row_id, line in numbered if not line.role]
    if not body:
        return f"{opening}."
    return f"{opening}; the main text runs from top to bottom: {_cite(body)}."


def _role_sentences(numbered: list[tuple[str, Line]], column: tuple[float, float], crop: Crop) -> list[str]:
    """One sentence per role and position: "A marginal note <Sv_Text_4> sits in the left margin."."""
    placed: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row_id, line in numbered:
        if line.role:
            placed[(line.role, _role_position(line, column, crop))].append(row_id)
    return [_role_sentence(ROLE_NAMES[role], where, ids) for (role, where), ids in placed.items()]


def _role_position(line: Line, column: tuple[float, float], crop: Crop) -> str:
    if line.role != "marginalia":
        return crop.place(line.middle_x, line.baseline_y)
    return "in the left margin" if line.middle_x < sum(column) / 2 else "in the right margin"


def _role_sentence(name: str, where: str, ids: list[str]) -> str:
    if len(ids) == 1:
        return f"A {name} {_cite(ids)} sits {where}."
    return f"{_count(len(ids)).capitalize()} {name} lines {_cite(ids)} sit {where}."


def _fragment_sentences(fragments: list[tuple[str, Line]], crop: Crop) -> list[str]:
    if not fragments:
        return []
    edges = {crop.cut_edge(line) for _, line in fragments}
    along = "the top and bottom edges" if len(edges) == 2 else f"the {edges.pop()} edge"
    return [
        f"Cut-off parts of further lines {_cite([row_id for row_id, _ in fragments])} show along {along}."
    ]


def _place(across: float, down: float) -> str:
    """Words for a position given as fractions of the crop's width and height."""
    side = _third(across, ("left", "center", "right"))
    level = _third(down, ("upper", "middle", "lower"))
    if side == "center":
        return {"upper": "at the top center", "middle": "in the middle", "lower": "at the bottom center"}[
            level
        ]
    if level == "middle":
        return f"on the {side} side"
    return f"in the {level} {side} corner"


def _third(fraction: float, names: tuple[str, str, str]) -> str:
    if fraction < 1 / 3:
        return names[0]
    if fraction > 2 / 3:
        return names[2]
    return names[1]


def _cite(ids: list[str]) -> str:
    return " ".join(f"<{row_id}>" for row_id in ids)


def _count(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


def sample_pages(collection_dir: Path, count: int, rng: random.Random) -> list[Page]:
    """Up to ``count`` random pages of a collection with a scan and transcribed text."""
    xml_paths = sorted(
        p for p in collection_dir.rglob("page/*.xml") if not p.name.startswith("._")
    )  # "._*": macOS
    rng.shuffle(xml_paths)
    pages: list[Page] = []
    for xml_path in xml_paths:
        if len(pages) == count:
            break
        page = _load_page(collection_dir, xml_path)
        if page is not None:
            pages.append(page)
    return pages


def _load_page(collection_dir: Path, xml_path: Path) -> Page | None:
    scan = _find_scan(xml_path)
    if scan is None:
        return None
    regions = parse_page(xml_path)
    if not regions:
        return None
    return Page(
        collection=collection_dir.name,
        path=xml_path.relative_to(collection_dir.parent),
        scan=scan,
        regions=regions,
    )


def _find_scan(xml_path: Path) -> Path | None:
    """The scan a PAGE XML in ``<volume>/page/`` describes: the same stem, one folder up."""
    candidates = (xml_path.parent.parent / f"{xml_path.stem}{suffix}" for suffix in SCAN_SUFFIXES)
    return next((path for path in candidates if path.is_file()), None)


def split_of(page: Page, val_frac: float) -> str:
    """The page's split, by its hash, so a page's crops never straddle train and val."""
    return "val" if int(page.id, 16) % 1000 < val_frac * 1000 else "train"


def write_crops(page: Page, split_dir: Path, settings: CropSettings) -> list[CropRecord]:
    """Cut a page into crops, save them as JPEGs in ``split_dir`` and return their metadata records."""
    with Image.open(page.scan) as img:
        scan = img.convert("RGB")
    records: list[CropRecord] = []
    for k, crop in enumerate(crops_for_page(page.regions, scan.size, settings)):
        left, top, right, bottom = (round(v) for v in crop.rect)
        tile = scan.crop((left, top, right, bottom)).resize(crop.size, Image.Resampling.LANCZOS)
        name = f"{page.collection}_{page.id}_{k:02d}.jpg"
        tile.save(split_dir / name, quality=95)
        look = scan_look(tile)
        caption = caption_for(crop, page.provenance, look)
        records.append(
            CropRecord(
                file_name=name,
                caption=caption.text,
                collection=page.collection,
                page=str(page.path),
                width=crop.size[0],
                height=crop.size[1],
                **look.model_dump(),
                layout=caption.layout,
                lines=caption.lines,
            )
        )
    return records


def _create_metadata(split_dir: Path) -> TextIO:
    split_dir.mkdir(parents=True, exist_ok=True)
    return open(split_dir / "metadata.jsonl", "w", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt-root", required=True, help="GT folder with one subfolder per collection")
    ap.add_argument("--collections", nargs="*", default=None, help="collection names (default: all)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pages", type=int, default=40, help="pages sampled per collection")
    ap.add_argument("--val-frac", type=float, default=0.05, help="pages held out (by page hash)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--line-spacing", type=float, help="target line pitch in px")
    ap.add_argument("--min-line-spacing", type=float, help="skip crops scaled below this pitch")
    ap.add_argument(
        "--min-merge-spacing", type=float, help="smallest pitch a margin note may shrink its column to"
    )
    ap.add_argument(
        "--min-width", type=int, help="narrower regions are only cropped together with a wide one"
    )
    ap.add_argument("--min-lines", type=int)
    ap.add_argument("--overlap-lines", type=int, help="lines shared by consecutive windows of a column")
    ap.set_defaults(**CropSettings().model_dump())
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    settings = CropSettings.model_validate(vars(args))
    root = Path(args.gt_root).expanduser()
    out = Path(args.out_dir)
    rng = random.Random(args.seed)
    totals: Counter[str] = Counter()
    with ExitStack() as stack:
        metadata = {split: stack.enter_context(_create_metadata(out / split)) for split in SPLITS}
        for collection in args.collections or sorted(p.name for p in root.iterdir() if p.is_dir()):
            for page in sample_pages(root / collection, args.max_pages, rng):
                split = split_of(page, args.val_frac)
                records = write_crops(page, out / split, settings)
                metadata[split].writelines(f"{record.model_dump_json()}\n" for record in records)
                totals.update(pages=1, crops=len(records), lines=sum(len(record.lines) for record in records))
            print(f"{collection}: {dict(totals)}", flush=True)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
