"""A page for marking, by hand, where an opening's fold is: the yardstick for every cutter (CURATION task 1, E26).

200 scans, 40 from each of four kinds where the cut is hard and 40 easier ones, the pools and the ground truth
alike in each kind, drawn once from scans.lance (seed 0)
and kept in fold_marks_sample.lance. Each scan is shown whole and as a strip around the middle at higher resolution;
a click on the strip marks the fold, a click on the whole scan moves the strip. No cutter's answer is drawn, so the
mark is not led by one. Every mark is appended to fold_marks.lance (the last row of an image is its mark).

    uv run python runs/htr/curate/prototypes/mark_folds.py [random]   # then open http://localhost:7861

With ``random``: 150 described scans drawn at random instead (fold_marks_sample_random.lance), for rates that hold
for the pools.
"""
import sys
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr
import lance
import pandas as pd
import pyarrow as pa
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); SAMPLE = HOME / "fold_marks_sample.lance"; MARKS = HOME / "fold_marks.lance"
EACH, HALF, TALL, WIDE = 40, 0.11, 1500, 1000  # scans a kind; half the strip's width as a share of the scan; pixel sizes shown
KINDS = {"single": "No cut: a single page", "no_cut": "Must not be cut: one table or text across the fold", "unsure": "Cannot tell"}
MARK = pa.schema([("image", pa.utf8()), ("mark", pa.utf8()), ("x", pa.float32()), ("marked_at", pa.timestamp("s", "UTC"))])


def draw_sample() -> pd.DataFrame:
    """The scans to mark, hard kinds first; a scan belongs to the first kind it fits."""
    columns = ["image", "pool", "gutter_x", "gutter_score", "vl_layout", "vl_writing_at_fold", "vl_table_across_fold", "fold_placed", "fold_net", "fold_stands"]
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=columns, filter="vl_layout IS NOT NULL OR fold_net IS NOT NULL").to_pandas()
    said_opening = scans.vl_layout.isin(["opening_both_pages_written", "opening_one_page_empty"])
    opening = said_opening | (scans.vl_layout.isna() & (scans.gutter_score >= 0.5))  # not described: the gutter network's word
    kinds = [("cutters disagree", EACH, opening & ((scans.fold_placed - scans.fold_net).abs() > 0.01)),
             ("table across the fold", EACH, scans.vl_table_across_fold.fillna(False).astype(bool)),
             ("no fold line found", EACH, opening & (scans.fold_stands < 4.0)),
             ("writing at the fold", EACH, said_opening & scans.vl_writing_at_fold.fillna(False).astype(bool)),
             ("single page the network cuts", EACH // 2, (scans.vl_layout == "single_page") & (scans.gutter_score >= 0.5)),
             ("plain opening", EACH // 2, said_opening)]
    taken, parts = pd.Series(False, index=scans.index), []
    for name, count, fits in kinds:
        left = scans[fits & ~taken]; pools = sorted(left.pool.unique())  # the pools and the GT alike, as far as each has scans
        part = pd.concat(left[left.pool == pool].sample(min((count + i) // len(pools), (left.pool == pool).sum()), random_state=0) for i, pool in enumerate(pools))
        taken[part.index] = True; parts.append(part.assign(stratum=name))
    return pd.concat(parts).sample(frac=1, random_state=0)[["image", "pool", "stratum", "gutter_x"]].reset_index(drop=True)


def draw_random(count: int = 150) -> pd.DataFrame:
    """Described scans drawn at random, a third from each pool and the ground truth, none marked before: for rates
    that hold for the pools, which the hard kinds of the first 200 do not give."""
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "pool", "gutter_x"], filter="vl_layout IS NOT NULL").to_pandas()
    scans = scans[~scans.image.isin(set(lance.dataset(str(SAMPLE)).to_table(columns=["image"]).column("image").to_pylist()))]
    parts = [scans[scans.pool == pool].sample(count // 3, random_state=1) for pool in sorted(scans.pool.unique())]
    return pd.concat(parts).sample(frac=1, random_state=1).assign(stratum="random")[["image", "pool", "stratum", "gutter_x"]].reset_index(drop=True)


if not SAMPLE.exists():
    lance.write_dataset(pa.Table.from_pandas(draw_sample(), preserve_index=False), str(SAMPLE))
def draw_targeted(each: int = 75) -> pd.DataFrame:
    """Scans where marks are thin: those the first form calls a single page or a table across the fold (most are not
    to be cut: examples for a don't-cut model, and how often the form is wrong), and openings where the placer finds
    no fold line (marks for a trained placer); the pools and the ground truth alike, none marked before."""
    scans = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "pool", "gutter_x", "vl_layout", "vl_table_across_fold"], filter="vl_layout IS NOT NULL").to_pandas()
    lost = set(lance.dataset(str(HOME / "folds.lance")).to_table(columns=["image"], filter="fold_found = false").column("image").to_pylist())
    seen = set().union(*(set(lance.dataset(str(path)).to_table(columns=["image"]).column("image").to_pylist()) for path in (SAMPLE, RANDOM) if path.exists()))
    scans = scans[~scans.image.isin(seen)]
    kinds = [("first form: table across the fold", scans.vl_table_across_fold.fillna(False).astype(bool)),
             ("first form: single page", scans.vl_layout == "single_page"), ("no fold line found", scans.image.isin(lost))]
    taken, parts = pd.Series(False, index=scans.index), []
    for name, fits in kinds:
        left = scans[fits & ~taken]; pools = sorted(left.pool.unique())
        part = pd.concat(left[left.pool == pool].sample(min((each + i) // len(pools), (left.pool == pool).sum()), random_state=2) for i, pool in enumerate(pools))
        taken[part.index] = True; parts.append(part.assign(stratum=name))
    return pd.concat(parts).sample(frac=1, random_state=2)[["image", "pool", "stratum", "gutter_x"]].reset_index(drop=True)


RANDOM = HOME / "fold_marks_sample_random.lance"; TARGETED = HOME / "fold_marks_sample_targeted.lance"
if "random" in sys.argv[1:]:
    if not RANDOM.exists(): lance.write_dataset(pa.Table.from_pandas(draw_random(), preserve_index=False), str(RANDOM))
    sample = lance.dataset(str(RANDOM)).to_table().to_pandas()
elif "targeted" in sys.argv[1:]:
    if not TARGETED.exists(): lance.write_dataset(pa.Table.from_pandas(draw_targeted(), preserve_index=False), str(TARGETED))
    sample = lance.dataset(str(TARGETED)).to_table().to_pandas()
else:
    sample = lance.dataset(str(SAMPLE)).to_table().to_pandas()


def marks() -> dict[str, tuple[str, float | None]]:
    if not MARKS.exists(): return {}
    rows = lance.dataset(str(MARKS)).to_table().to_pandas().sort_values("marked_at")
    return {row.image: (row.mark, None if pd.isna(row.x) else float(row.x)) for row in rows.itertuples()}


def save(image: str, mark: str, x: float | None) -> None:
    row = pa.table({"image": [image], "mark": [mark], "x": [x], "marked_at": [datetime.now(UTC).replace(microsecond=0)]}, schema=MARK)
    lance.write_dataset(row, str(MARKS), mode="append" if MARKS.exists() else "create")


def views(at: int, centre: float | None) -> tuple[Image.Image, Image.Image, float, str]:
    """The whole scan and the strip around ``centre`` (the network's gutter when None), the scan's mark drawn on both."""
    row = sample.iloc[at]; mark, x = marks().get(row.image, (None, None))
    centre = float(centre if centre is not None else x if x is not None else row.gutter_x if 0.2 < row.gutter_x < 0.8 else 0.5)
    left, right = max(0.0, centre - HALF), min(1.0, centre + HALF)
    with Image.open(row.image) as scan:
        scan = scan.convert("RGB"); w, h = scan.size
        strip = scan.crop((int(left * w), 0, int(right * w), h)); strip = strip.resize((max(1, round(strip.width * TALL / h)), TALL))
        whole = scan.resize((WIDE, max(1, round(h * WIDE / w))))
    pen = ImageDraw.Draw(whole); pen.rectangle((left * WIDE, 0, right * WIDE, whole.height - 1), outline=(42, 120, 214), width=3)
    if x is not None:
        pen.line((x * WIDE, 0, x * WIDE, whole.height), fill=(235, 30, 30), width=2)
        if left <= x <= right:
            at_x = (x - left) / (right - left) * strip.width; ImageDraw.Draw(strip).line((at_x, 0, at_x, TALL), fill=(235, 30, 30), width=2)
    done = len(set(marks()) & set(sample.image))
    said = "not marked" if mark is None else f"fold at {x:.4f} of the width" if mark == "fold" else KINDS[mark]
    return whole, strip, centre, f"**Scan {at + 1} of {len(sample)}** · {done} marked · this scan: **{said}** · `{Path(row.image).name}` ({row.pool})"


def mark_fold(at: int, centre: float, event: gr.SelectData):
    left, right = max(0.0, centre - HALF), min(1.0, centre + HALF)
    with Image.open(sample.iloc[at].image) as scan: shown = max(1, round((right - left) * scan.width * TALL / scan.height))
    save(sample.iloc[at].image, "fold", left + event.index[0] / shown * (right - left))
    return views(at, centre)


def move_strip(at: int, event: gr.SelectData): return views(at, event.index[0] / WIDE)
def mark_kind(at: int, kind: str): save(sample.iloc[at].image, kind, None); return step(at, 1)
def step(at: int, by: int): at = min(len(sample) - 1, max(0, at + by)); return (at, *views(at, None))


def first_unmarked() -> int:
    held = marks(); return next((i for i, image in enumerate(sample.image) if image not in held), 0)


with gr.Blocks(title="Mark the fold") as page:
    at, centre = gr.State(0), gr.State(0.5)
    gr.Markdown("### Mark the fold\nClick on the **strip** (right) exactly where the scan should be cut into two pages. A click on the **whole scan** (left) moves the strip there. Click again to correct; then **Next**.")
    status = gr.Markdown()
    with gr.Row():
        back, onward = gr.Button("← Back"), gr.Button("Next →", variant="primary")
        buttons = {kind: gr.Button(label) for kind, label in KINDS.items()}
    with gr.Row():
        whole = gr.Image(label="Whole scan: click to move the strip", interactive=False, type="pil", scale=3)
        strip = gr.Image(label="Strip: click on the fold", interactive=False, type="pil", scale=2)
    shown = [whole, strip, centre, status]
    strip.select(mark_fold, [at, centre], shown); whole.select(move_strip, [at], shown)
    back.click(lambda i: step(i, -1), [at], [at, *shown]); onward.click(lambda i: step(i, 1), [at], [at, *shown])
    for kind, button in buttons.items(): button.click(lambda i, kind=kind: mark_kind(i, kind), [at], [at, *shown])
    page.load(lambda: step(first_unmarked(), 0), None, [at, *shown])

if __name__ == "__main__":
    page.launch(server_name="127.0.0.1", server_port=7861, allowed_paths=[])
