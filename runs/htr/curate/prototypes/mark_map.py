"""Marking scans in bulk on the map of their embeddings (CURATION E35), after marimo's bulk-labelling example.

The 14,985 scans with a Qwen3-VL-Embedding 8B vector, laid out by UMAP (scan_map.lance) and coloured by cluster,
by the vision-language model's answer or by the marks made so far. Drag a box over a region of the map: twenty of
its scans are shown, and one button marks every scan in the box. Marks go to region_marks.lance (image, mark,
marked_at); the last row of an image is its mark.

    uv run --with marimo --with altair marimo run runs/htr/curate/prototypes/mark_map.py --host 127.0.0.1 --port 7863 --headless --no-token
"""
import marimo

app = marimo.App(width="full")


@app.cell
def _():
    from datetime import UTC, datetime
    from hashlib import sha1
    from pathlib import Path

    import altair as alt
    import lance
    import marimo as mo
    import pandas as pd
    import pyarrow as pa
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    HOME = Path("runs/htr/curate")
    MARKS = HOME / "region_marks.lance"
    THUMBS = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs")
    ANSWERS = {"cut": "Cut", "no_cut": "Do not cut", "single": "Single page", "not_a_page": "Not a page"}
    ROW = pa.schema([("image", pa.utf8()), ("mark", pa.utf8()), ("marked_at", pa.timestamp("s", "UTC"))])
    alt.data_transformers.disable_max_rows()
    return ANSWERS, HOME, Image, MARKS, ROW, THUMBS, UTC, alt, datetime, lance, mo, pa, pd, sha1


@app.cell
def _(HOME, lance):
    _map = lance.dataset(str(HOME / "scan_map.lance")).to_table().to_pandas()
    _said = lance.dataset(str(HOME / "scans.lance")).to_table(
        columns=["image", "pool", "vl_layout", "vl_writing_layout", "vl_table_across_fold"], filter="vl_kind IS NOT NULL"
    ).to_pandas()
    scans = _map.merge(_said, on="image", how="left")
    scans["cluster"] = scans.cluster.astype(str)
    scans["said"] = scans.vl_layout.where(~scans.vl_table_across_fold.fillna(False).astype(bool), "table across the fold")
    scans["at"] = range(len(scans))
    return (scans,)


@app.cell
def _(mo):
    get_saves, set_saves = mo.state(0)
    colour_by = mo.ui.dropdown(
        {"Cluster": "cluster", "The model's answer": "said", "Table or running text": "vl_writing_layout", "Pool": "pool", "Your marks": "mark"},
        value="The model's answer", label="Colour by",
    )
    return colour_by, get_saves, set_saves


@app.cell
def _(MARKS, get_saves, lance, scans):
    get_saves()
    if MARKS.exists():
        _rows = lance.dataset(str(MARKS)).to_table().to_pandas().sort_values("marked_at").drop_duplicates("image", keep="last")
        marked = dict(zip(_rows.image, _rows.mark))
    else:
        marked = {}
    points = scans.assign(mark=scans.image.map(marked).fillna("not marked"))
    return marked, points


@app.cell
def _(alt, colour_by, mo, points):
    _chart = (
        alt.Chart(points[["at", "map_x", "map_y", "cluster", "said", "vl_writing_layout", "pool", "mark"]])
        .mark_circle(size=9, opacity=0.6)
        .encode(x=alt.X("map_x:Q", axis=None), y=alt.Y("map_y:Q", axis=None), color=alt.Color(f"{colour_by.value}:N", legend=alt.Legend(columns=2, symbolLimit=60)))
        .properties(width=820, height=640)
    )
    chart = mo.ui.altair_chart(_chart, chart_selection="interval", legend_selection=False)
    return (chart,)


@app.cell
def _(Image, THUMBS, sha1):
    def thumb(image: str) -> str:
        path = THUMBS / f"{sha1(image.encode()).hexdigest()}.jpg"
        if not path.exists():
            THUMBS.mkdir(parents=True, exist_ok=True)
            with Image.open(image) as scan:
                scan.draft("RGB", (1280, 1280))
                scan = scan.convert("RGB")
                scan.thumbnail((640, 640))
                scan.save(path, "JPEG", quality=82)
        return str(path)

    return (thumb,)


@app.cell
def _(chart, points):
    _picked = chart.value
    selected = points[points["at"].isin(_picked["at"])] if _picked is not None and len(_picked) and len(_picked) < len(points) else points.iloc[0:0]
    return (selected,)


@app.cell
def _(ANSWERS, MARKS, ROW, UTC, datetime, lance, mo, pa, selected, set_saves):
    def save(mark: str) -> None:
        if not len(selected):
            return
        now = datetime.now(UTC).replace(microsecond=0)
        table = pa.table({"image": selected.image.tolist(), "mark": [mark] * len(selected), "marked_at": [now] * len(selected)}, schema=ROW)
        lance.write_dataset(table, str(MARKS), mode="append" if MARKS.exists() else "create")
        set_saves(lambda count: count + 1)

    buttons = mo.hstack(
        [mo.ui.button(label=f"Mark all {len(selected)} as: {label}", on_click=lambda _value, mark=mark: save(mark), disabled=not len(selected)) for mark, label in ANSWERS.items()],
        justify="start",
    )
    return (buttons,)


@app.cell
def _(mo, selected, thumb):
    if len(selected):
        _shown = selected.sample(min(20, len(selected)), random_state=0)
        _mix = ", ".join(f"{name} {count}" for name, count in selected.said.value_counts().head(5).items())
        gallery = mo.vstack([
            mo.md(f"**{len(selected)} scans in the box.** The model's answers: {_mix}. Twenty of them, drawn at random:"),
            mo.hstack([mo.image(src=thumb(image), width=230) for image in _shown.image], wrap=True, justify="start"),
        ])
    else:
        gallery = mo.md("Drag a box over a part of the map to see its scans.")
    return (gallery,)


@app.cell
def _(buttons, chart, colour_by, gallery, marked, mo, points):
    mo.vstack([
        mo.md(f"### Mark scans by region of the map\n{len(points):,} scans, each a dot; scans that look alike lie together. Drag a box, look at the twenty shown, and if they are all one answer mark the whole box. **{len(marked):,} marked so far.**"),
        colour_by,
        mo.hstack([chart, mo.vstack([buttons, gallery])], widths=[1, 1], align="start"),
    ])
    return


if __name__ == "__main__":
    app.run()
