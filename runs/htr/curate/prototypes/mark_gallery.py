"""A gallery for marking in bulk whether a scan is to be cut (CURATION E34).

Scans are shown twenty at a time, each screen holding scans the first form gives one answer (a table across the fold,
a single page, an opening), every scan starting at that guess. An answer is picked (cut, must not be cut, single page, not a page) and a click
on a scan gives it that answer; "Save and next" stores the twenty marks. Only the wrong ones need a click.
Marks go to cut_marks.lance (image, mark, guess, marked_at); the sample, drawn once, is cut_marks_sample.lance.

    uv run python runs/htr/curate/prototypes/mark_gallery.py     # then open http://localhost:7862
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path

import gradio as gr
import lance
import pandas as pd
import pyarrow as pa
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); SAMPLE = HOME / "cut_marks_sample.lance"; MARKS = HOME / "cut_marks.lance"
THUMBS = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs"); WIDE, SCREEN = 640, 20
ANSWERS = ["cut", "no_cut", "single", "not_a_page"]
SAYS = {"cut": "CUT: two pages, each reads alone", "no_cut": "DO NOT CUT: one table or text across the fold", "single": "SINGLE PAGE", "not_a_page": "NOT A PAGE"}
COLOUR = {"cut": (27, 175, 122), "no_cut": (235, 104, 52), "single": (42, 120, 214), "not_a_page": (140, 140, 140)}
ROW = pa.schema([("image", pa.utf8()), ("mark", pa.utf8()), ("guess", pa.utf8()), ("marked_at", pa.timestamp("s", "UTC"))])


def draw_sample() -> pd.DataFrame:
    """1,200 described scans: every kind the first form says is not to be cut, as far as there are scans, and as many
    openings; the pools and the ground truth alike; scans marked on the fold page left out. Screens of one guess."""
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "pool", "vl_layout", "vl_table_across_fold"], filter="vl_layout IS NOT NULL").to_pandas()
    seen = set(lance.dataset(str(HOME / "fold_marks.lance")).to_table(columns=["image"]).column("image").to_pylist())
    scans = scans[~scans.image.isin(seen)]
    kinds = [("no_cut", 300, scans.vl_table_across_fold.fillna(False).astype(bool)), ("single", 400, scans.vl_layout == "single_page"),
             ("not_a_page", 60, scans.vl_layout.isin(["not_a_page", "several_loose_sheets"])), ("cut", 440, scans.vl_layout.isin(["opening_both_pages_written", "opening_one_page_empty"]))]
    taken, parts = pd.Series(False, index=scans.index), []
    for guess, count, fits in kinds:
        left = scans[fits & ~taken]; pools = sorted(left.pool.unique())
        part = pd.concat(left[left.pool == pool].sample(min((count + i) // len(pools), (left.pool == pool).sum()), random_state=3) for i, pool in enumerate(pools))
        taken[part.index] = True; parts.append(part.sample(frac=1, random_state=3).assign(guess=guess))
    return pd.concat(parts)[["image", "pool", "guess"]].reset_index(drop=True)


def thumb(image: str) -> Path:
    path = THUMBS / f"{sha1(image.encode()).hexdigest()}.jpg"
    if not path.exists():
        with Image.open(image) as scan:
            scan.draft("RGB", (WIDE * 2, WIDE * 2)); scan = scan.convert("RGB"); scan.thumbnail((WIDE, WIDE)); scan.save(path, "JPEG", quality=82)
    return path


if not SAMPLE.exists():
    lance.write_dataset(pa.Table.from_pandas(draw_sample(), preserve_index=False), str(SAMPLE))
sample = lance.dataset(str(SAMPLE)).to_table().to_pandas()
THUMBS.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(24) as pool: list(pool.map(thumb, sample.image))
screens = [list(range(start, min(start + SCREEN, len(sample)))) for start in range(0, len(sample), SCREEN)]


def held() -> dict[str, str]:
    if not MARKS.exists(): return {}
    rows = lance.dataset(str(MARKS)).to_table().to_pandas().sort_values("marked_at")
    return dict(zip(rows.image, rows.mark))


def shown(screen: int, marks: list[str]):
    """The screen's scans, each framed in its mark's colour with the mark written under it."""
    cards = []
    for at, mark in zip(screens[screen], marks):
        card = Image.open(thumb(sample.image[at])).convert("RGB"); pen = ImageDraw.Draw(card)
        pen.rectangle((0, 0, card.width - 1, card.height - 1), outline=COLOUR[mark], width=14)
        cards.append((card, SAYS[mark].split(":")[0]))
    done = len(set(held()) & set(sample.image)); guess = sample.guess[screens[screen][0]]
    return cards, f"**Screen {screen + 1} of {len(screens)}** · {done} of {len(sample)} scans saved · these start as **{SAYS[guess]}**: click the ones that are something else"


def start(screen: int):
    saved = held(); marks = [saved.get(sample.image[at], sample.guess[at]) for at in screens[screen]]
    return (screen, marks, *shown(screen, marks))


def paint(screen: int, marks: list[str], brush: str, event: gr.SelectData):
    """The clicked scan takes the brush's answer; clicked again with the same brush, it goes back to the guess."""
    marks = list(marks); at = event.index
    marks[at] = sample.guess[screens[screen][at]] if marks[at] == brush else brush
    return (marks, *shown(screen, marks))


def save_and_go(screen: int, marks: list[str], by: int):
    now = datetime.now(UTC).replace(microsecond=0); rows = screens[screen]
    table = pa.table({"image": [sample.image[at] for at in rows], "mark": marks, "guess": [sample.guess[at] for at in rows], "marked_at": [now] * len(rows)}, schema=ROW)
    lance.write_dataset(table, str(MARKS), mode="append" if MARKS.exists() else "create")
    return start(min(len(screens) - 1, max(0, screen + by)))


def first_unsaved() -> int:
    saved = held(); return next((i for i, rows in enumerate(screens) if sample.image[rows[0]] not in saved), 0)


with gr.Blocks(title="Cut or not") as page:
    screen, marks = gr.State(0), gr.State([])
    gr.Markdown("### Cut or not, in bulk\nEvery scan starts at the system's guess, shown by its frame: green CUT, orange DO NOT CUT, blue SINGLE PAGE, grey NOT A PAGE. **Pick the right answer below, then click every scan that should have it** (click again to undo). Then **Save and next**. A scan is *cut* when it shows two pages that each read on their own; *do not cut* when one table, ledger row or text runs across the fold.")
    status = gr.Markdown()
    with gr.Row():
        brush = gr.Radio([(SAYS[answer], answer) for answer in ANSWERS], value="cut", label="A click sets the scan to", scale=4)
        back, onward = gr.Button("← Save and back", scale=1), gr.Button("Save and next →", variant="primary", scale=1)
    gallery = gr.Gallery(columns=5, height=1500, allow_preview=False, object_fit="contain", show_label=False)
    gallery.select(paint, [screen, marks, brush], [marks, gallery, status])
    back.click(lambda s, m: save_and_go(s, m, -1), [screen, marks], [screen, marks, gallery, status])
    onward.click(lambda s, m: save_and_go(s, m, 1), [screen, marks], [screen, marks, gallery, status])
    page.load(lambda: start(first_unsaved()), None, [screen, marks, gallery, status])

if __name__ == "__main__":
    page.launch(server_name="127.0.0.1", server_port=7862)
