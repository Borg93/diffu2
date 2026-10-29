"""MOSS-Image's page pairs (diffu_page.data.page_pairs) built from a tiny ground truth of three volumes: each page's
prompts are the ones drawing compiles for it planned whole, from its sample or without, its sample is a page of its own
volume sharing none of its lines, whole volumes are held out, the build records what became of every scan, a stopped
build resumes from what it recorded, and diffu-moss-cache-pairs caches the pairs as their rows give them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from diffusers.modular_pipelines import SequentialPipelineBlocks
from PIL import Image, ImageDraw

from diffu_moss.train.cache import CacheSettings, load_sample, read_entries, write_cache
from diffu_moss.train.pairs import PagePairRow, Variant
from diffu_page.captions.look import scan_look
from diffu_page.captions.prompt import PageStyle
from diffu_page.captions.provenance import provenance_of
from diffu_page.captions.restyle import ReferenceCaption
from diffu_page.cli.generate import page_reference
from diffu_page.crops.grid import as_reference
from diffu_page.data.page_pairs import (
    BuildCounts,
    BuildRecord,
    BuildSettings,
    ScanResult,
    ScanSettings,
    build,
    digest,
)
from diffu_page.evalset import read_crops, read_jsonl
from diffu_page.layout import load_xml_layout, scan_pages
from diffu_page.pipeline import FAMILIES, crop_blocks
from diffu_page.plan import CropPlan, plan_whole_page

ARJEPLOG = "GT-RA_arjeplogs_tingslag_20260220"
EARLY, LATE = "Arjeplogs tingslag (1690-1700)", "Arjeplogs tingslag (1701-1710)"
DEPUTERADE, DEPUTERADE_VOLUME = "GT-SSA_deputerade_20260218", "SSA_Deputerade"
PIXELS = 128 * 128  # small pages: each page's sides on the 16-px grid, its scan cut and prompted as a full one
PAPER, INK = (222, 205, 170), (70, 50, 30)
LONG_PAGE_LINES = 150  # lines of a page whose prompt passes the text encoder's 4,096 tokens


def _line(line_id: str, box: tuple[int, int, int, int], text: str) -> str:
    x0, y0, x1, y1 = box
    return (
        f'<TextLine id="{line_id}"><Coords points="{x0},{y0} {x1},{y0} {x1},{y1} {x0},{y1}"/>'
        f'<Baseline points="{x0},{y1 - 10} {x1},{y1 - 10}"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></TextLine>'
    )


def _write_page(
    pages: Path, name: str, size: tuple[int, int], columns: list[tuple[int, int, list[str]]], *, scan: bool = True
) -> None:
    """A PAGE file ``pages/<name>.xml`` of ``size`` px, a text region per column (its left and right x and its lines'
    texts, at a 40 px pitch from y 100) and, with ``scan``, its scan beside the ``page/`` dir: yellowed paper, each line a
    dark brown bar."""
    regions, bars = [], []
    for k, (x0, x1, texts) in enumerate(columns):
        boxes = [(x0, 100 + 40 * i, x1, 130 + 40 * i) for i in range(len(texts))]
        lines = "".join(_line(f"r{k}l{i}", box, text) for i, (box, text) in enumerate(zip(boxes, texts, strict=True)))
        regions.append(f'<TextRegion id="r{k}" type="paragraph">{lines}</TextRegion>')
        bars += boxes
    pages.mkdir(parents=True, exist_ok=True)
    (pages / f"{name}.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
        f'<Page imageFilename="{name}.png" imageWidth="{size[0]}" imageHeight="{size[1]}">{"".join(regions)}</Page>'
        "</PcGts>",
        encoding="utf-8",
    )
    if scan:
        image = Image.new("RGB", size, PAPER)
        draw = ImageDraw.Draw(image)
        for box in bars:
            draw.rectangle(box, fill=INK)
        image.save(pages.parent / f"{name}.png")


def _texts(tag: str, count: int) -> list[str]:
    return [f"{tag} rad {i} om tinget" for i in range(count)]


@pytest.fixture(scope="module")
def tiny_gt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny GT of two collections, three volumes and ten PAGE files (and a macOS ``._`` file):

    - Arjeplogs tingslag (1690-1700): an opening (two pages: three lines left, two right), a single page, a PAGE file
      without its scan, one that is not XML (its scan beside it);
    - Arjeplogs tingslag (1701-1710): two single pages sharing a line, so neither has a sample;
    - SSA Deputerade: a page whose prompt is too long, two single pages, a page without a line.
    """
    root = tmp_path_factory.mktemp("gt")
    early = root / ARJEPLOG / EARLY / EARLY / "page"
    _write_page(early, "0001", (1600, 1000), [(100, 700, _texts("vänster", 3)), (900, 1500, _texts("höger", 2))])
    _write_page(early, "0002", (800, 1000), [(100, 700, _texts("enkel", 3))])
    _write_page(early, "0003", (800, 1000), [(100, 700, _texts("utan bild", 2))], scan=False)
    (early / "0004.xml").write_text("<PcGts><Page", encoding="utf-8")
    Image.new("RGB", (800, 1000), PAPER).save(early.parent / "0004.png")
    (early / "._0001.xml").write_bytes(b"\x00\x05\x16\x07 macOS resource fork")
    late = root / ARJEPLOG / LATE / LATE / "page"
    _write_page(late, "0001", (800, 1000), [(100, 700, ["Anno 1701", *_texts("första", 2)])])
    _write_page(late, "0002", (800, 1000), [(100, 700, ["Anno 1701", *_texts("andra", 2)])])
    deputerade = root / DEPUTERADE / DEPUTERADE_VOLUME / DEPUTERADE_VOLUME / "page"
    long_texts = [
        f"rad {i} ur deputeradenas protokoll om stadens ärenden och borgerskapets klagomål" for i in range(150)
    ]
    _write_page(deputerade, "0001", (800, 6200), [(100, 700, long_texts)])
    _write_page(deputerade, "0002", (800, 1000), [(100, 700, _texts("deputerade", 3))])
    _write_page(deputerade, "0003", (800, 1000), [(100, 700, _texts("magistraten", 2))])
    _write_page(deputerade, "0004", (800, 1000), [])
    return root


def _settings(gt: Path, model: Path, **changes: object) -> BuildSettings:
    settings = BuildSettings(
        scan=ScanSettings(gt_root=str(gt), model=str(model), pixels=PIXELS), val_volumes=0.34, eval_pages=2
    )
    return settings.model_copy(update=changes)


@pytest.fixture(scope="module")
def built(tiny_gt: Path, tiny_moss: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny GT's page pairs, built on two processes."""
    out = tmp_path_factory.mktemp("page_pairs")
    build(out, _settings(tiny_gt, tiny_moss), workers=2)
    return out


def _rows(out: Path, manifest: str) -> list[PagePairRow]:
    return read_jsonl(out / manifest, PagePairRow)


def _all_rows(out: Path) -> list[PagePairRow]:
    return _rows(out, "train.jsonl") + _rows(out, "val.jsonl")


def _drawing_prompt(plan: CropPlan, sample: Image.Image | None) -> str:
    """The prompt the page pipeline's MOSS steps give ``plan`` drawn with ``sample`` (or without one)."""
    blocks = crop_blocks(FAMILIES["MossAutoBlocks"]).sub_blocks
    steps = SequentialPipelineBlocks.from_blocks_dict({name: blocks[name] for name in ("crop_prompt", "layout_prompt")})
    return steps.init_pipeline()(crop_plan=plan, image=sample, output="prompt")


def test_page_pairs_prompt_a_page_as_drawing_prompts_it_planned_whole_from_its_sample_or_without(
    tiny_gt: Path, built: Path
) -> None:
    xml = tiny_gt / ARJEPLOG / EARLY / EARLY / "page" / "0001.xml"
    right = scan_pages(load_xml_layout(xml))[1]
    row = next(row for row in _all_rows(built) if row.id == f"{digest(xml.relative_to(tiny_gt).as_posix())}_right")
    with Image.open(built / row.target) as page:
        style = PageStyle(provenance=provenance_of(ARJEPLOG, xml), look=scan_look(page))
    sample = built / row.reference

    def drawn(form: ReferenceCaption | None) -> str:
        reference = page_reference(sample, form) if form else None
        plan = plan_whole_page(right.layout, style, pixels=PIXELS, reference=reference)
        return _drawing_prompt(plan, Image.open(sample) if form else None)

    expected = {
        Variant.NO_REFERENCE: drawn(None),
        Variant.LOOK: drawn(ReferenceCaption.NAMED),
        Variant.PLAIN: drawn(ReferenceCaption.HAND),
    }
    assert row.prompts == expected
    assert [json.loads(row.prompts[variant])["summary"]["caption"] for variant in Variant] == [
        "Flat, evenly lit archival scan of a handwritten page from the court minutes of Arjeplogs tingslag, dated "
        "1690–1700. The yellowed paper carries 2 lines of cursive handwriting in dark brown ink.",
        "Create a flat, evenly lit archival scan of a new handwritten page from the court minutes of Arjeplogs "
        "tingslag, dated 1690–1700, in the hand, ink and paper of the sample image. The yellowed paper carries 2 lines "
        "of cursive handwriting in dark brown ink.",
        "Create a flat, evenly lit archival scan of a new handwritten page from the court minutes of Arjeplogs "
        "tingslag, dated 1690–1700, in the hand, ink and paper of the sample image. The paper carries 2 lines of "
        "cursive handwriting.",
    ]
    plain_lines = [item for item in json.loads(row.prompts[Variant.PLAIN])["items"] if item["label"] == "text"]
    assert {item["caption"] for item in plain_lines} == {
        "Handwritten line, cursive script in the sample image's hand and ink."
    }


def test_page_pairs_cut_a_page_from_its_scan_at_its_whole_page_size_and_its_sample_at_the_reference_size(
    tiny_gt: Path, built: Path
) -> None:
    xml = tiny_gt / ARJEPLOG / EARLY / EARLY / "page" / "0001.xml"
    right = scan_pages(load_xml_layout(xml))[1]
    page_id = f"{digest(xml.relative_to(tiny_gt).as_posix())}_right"
    with Image.open(xml.parent.parent / "0001.png") as scan:
        cut = scan.convert("RGB").crop(right.rect)

    with Image.open(built / "pages" / f"{page_id}.jpg") as page, Image.open(built / "refs" / f"{page_id}.jpg") as ref:
        page_error = _mean_difference(page, cut.resize(page.size, Image.Resampling.LANCZOS))
        ref_error = _mean_difference(ref, as_reference(cut))
        sizes = page.size, ref.size

    assert right.rect == (800.0, 0.0, 1600.0, 1000.0)
    # 800 x 1000 px at about 128 x 128 px and at the reference's 0.5 MP, each side on the 16-px grid
    assert sizes == ((112, 144), (640, 800))
    assert page_error < 2  # the scan's own pixels, but for JPEG's
    assert ref_error < 2


def _mean_difference(image: Image.Image, expected: Image.Image) -> float:
    pairs = zip(image.convert("RGB").tobytes(), expected.convert("RGB").tobytes(), strict=True)
    return sum(abs(a - b) for a, b in pairs) / (3 * image.width * image.height)


def test_page_pairs_pair_each_page_with_another_page_of_its_volume_sharing_none_of_its_lines(built: Path) -> None:
    results = [ScanResult.model_validate_json(path.read_bytes()) for path in (built / "scanned").glob("*.json")]
    pages = {page.id: page for result in results for page in result.pages}

    rows = _all_rows(built)

    assert sorted(len(pages[row.id].lines) for row in rows) == [2, 2, 3, 3, 3]
    for row in rows:
        reference = pages[Path(row.reference).stem]
        assert (reference.volume, reference.id != row.id) == (row.volume, True)
        assert not pages[row.id].texts() & reference.texts()
    assert not any(LATE in row.volume for row in rows)  # its two pages share "Anno 1701"


def test_page_pairs_hold_out_whole_volumes_their_pairs_in_val_and_a_drawable_eval_set(built: Path) -> None:
    record = BuildRecord.model_validate_json((built / "build.json").read_text(encoding="utf-8"))

    held_out = record.held_out_volumes

    assert held_out == [f"{ARJEPLOG}/{EARLY}"]
    assert {row.volume for row in _rows(built, "val.jsonl")} == set(held_out)
    assert not {row.volume for row in _rows(built, "train.jsonl")} & set(held_out)
    eval_crops = read_crops(built / "val_eval")
    # --eval-pages 2 of its 3 pairs, spread over them
    assert [crop.file_name for crop in eval_crops] == [f"{row.id}.jpg" for row in _rows(built, "val.jsonl")[:2]]
    for crop in eval_crops:
        assert crop.reference is not None
        assert (built / "val_eval" / crop.file_name).samefile(built / "pages" / crop.file_name)
        assert (built / "val_eval" / crop.reference).samefile(built / crop.reference)


def test_page_pairs_record_what_became_of_every_scan_and_page(built: Path) -> None:
    record = BuildRecord.model_validate_json((built / "build.json").read_text(encoding="utf-8"))

    counts = record.counts

    assert counts == BuildCounts(
        scans=10,  # the macOS ._ file is not read
        no_scan=1,
        no_text=1,
        failed=1,
        pages=8,  # the opening's two, a single page and the 2 + 3 of the other volumes
        too_long=1,
        no_partner=2,
        train_pairs=2,
        val_pairs=3,
        eval_pages=2,
    )
    assert list(record.failed) == [f"{ARJEPLOG}/{EARLY}/{EARLY}/page/0004.xml"]


def test_page_pairs_resume_a_stopped_build_from_the_scans_it_recorded(
    tiny_gt: Path, tiny_moss: Path, tmp_path: Path
) -> None:
    gt = tmp_path / "gt"
    shutil.copytree(tiny_gt, gt)
    out = tmp_path / "out"
    build(out, _settings(gt, tiny_moss), workers=1)
    first = {path.name: path.read_bytes() for path in (out / "pages").iterdir()}
    for scan in gt.rglob("*.png"):  # read again, every page would come out black
        Image.new("RGB", Image.open(scan).size, "black").save(scan)

    build(out, _settings(gt, tiny_moss), workers=1)

    assert {path.name: path.read_bytes() for path in (out / "pages").iterdir()} == first


def test_page_pairs_refuse_to_resume_a_build_cut_at_another_size(tiny_gt: Path, tiny_moss: Path, built: Path) -> None:
    settings = _settings(tiny_gt, tiny_moss)
    larger = settings.model_copy(update={"scan": settings.scan.model_copy(update={"pixels": 2 * PIXELS})})

    with pytest.raises(ValueError, match="holds a build of gt_root="):
        build(built, larger, workers=1)


def test_cache_of_page_pairs_holds_each_rows_prompt_in_its_form_as_long_as_its_row_counts_it(
    built: Path, tiny_moss: Path, tmp_path: Path
) -> None:
    rows = {row.id: row for row in _rows(built, "train.jsonl")}

    write_cache(tiny_moss, built / "train.jsonl", tmp_path, CacheSettings(device="cpu", every_form=True))

    entries = read_entries(tmp_path)
    assert sorted((entry.pair, entry.variant) for entry in entries) == sorted(
        (pair, variant) for pair in rows for variant in Variant
    )
    for entry in entries:
        row = rows[entry.pair]
        assert entry.prompt == row.prompts[entry.variant]
        assert entry.tokens == row.tokens[entry.variant]
        assert entry.target_size == row.size
        assert (load_sample(tmp_path, entry, "cpu").reference is not None) is entry.variant.takes_reference
