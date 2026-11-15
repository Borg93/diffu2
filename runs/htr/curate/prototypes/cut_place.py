"""Placing an opening's cut at full resolution (CURATION E18).

The gutter network answers to 1 % of the scan's width. Here the cut is moved, within a band around the network's
answer, onto the fold itself: the band is read at about 2,400 px of scan width; thin dark vertical detail is brought
out (a closing along the row minus the image); a fold is dark down the whole height where writing is dark in some
rows only, so a column is scored by the detail it holds in most of its rows (a low quantile down the column), the
band sheared a little each way since a fold is seldom upright. Where no column stands out the network's answer is
kept.

First tried and dropped: the column with the least writing. It leaves the fold (itself a dark line) for the page's
inner margin and cut into labelled lines twice as often as the network (389 Svea scans: 0.21 against 0.11).

The labels are not used to place the cut. They score it: on scans with labelled lines on both pages, how often the
cut lies inside a labelled line's box, before and after, by how wide the gap between the two pages' text is.

    uv run python runs/htr/curate/prototypes/cut_place.py POOL SCANS [OUT.jsonl]
"""
import json, sys
from multiprocessing import Pool
import lance, numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage
from diffu_page.page.layout.xml_replay import load_xml_layout

Image.MAX_IMAGE_PIXELS = None
WIDTH = 2400      # the scan's width the band is read at
BAND = 0.03       # the band reaches this share of the width to either side of the network's answer
THIN = 31         # px at WIDTH: dark detail narrower than this along a row is a line or a stroke
MOST = 0.35       # a fold is dark in all but this share of a column's rows
LEANS = (-0.02, -0.01, 0.0, 0.01, 0.02)  # the fold's lean tried, px across per px down
CLEAR = 4.0       # a fold's column stands this many times above the band's usual column


def place(image: str, x: float) -> tuple[float, float] | None:
    """The cut for the scan at ``image`` whose gutter the network puts at ``x`` (a share of the width), and how far
    the fold found stands above the band's other columns (under :data:`CLEAR`: none found, the cut is ``x``)."""
    try:
        with Image.open(image) as scan:
            scan.draft("L", (WIDTH, WIDTH * 4)); grey = scan.convert("L")
    except (OSError, ValueError):
        return None
    k = WIDTH / grey.width
    if k < 1: grey = grey.resize((WIDTH, max(1, round(grey.height * k))), Image.Resampling.BILINEAR)
    w = grey.width; left, right = max(0, round((x - BAND) * w)), min(w, round((x + BAND) * w))
    band = np.asarray(grey.crop((left, 0, right, grey.height)), dtype=np.float32)
    band = band[band.shape[0] // 8: band.shape[0] - band.shape[0] // 8]  # the middle of the height: a page's top and bottom edges differ
    thin = ndimage.grey_closing(band, size=(1, THIN)) - band
    rows = np.arange(len(thin)) - len(thin) / 2; best = (0.0, x)
    for lean in LEANS:
        shift = np.round(lean * rows).astype(int)
        sheared = np.stack([np.roll(row, -move) for row, move in zip(thin, shift)])
        column = np.quantile(sheared, MOST, axis=0); column[:THIN // 2] = 0; column[-(THIN // 2):] = 0
        at = int(np.argmax(column)); stands = float(column[at] / (np.median(column) + 2.0))
        if stands > best[0]: best = (stands, (left + at) / w)
    return (best[1], best[0]) if best[0] >= CLEAR else (x, best[0])


def judge(row: tuple[str, str, float]) -> dict | None:
    image, labels, x = row
    try: layout = load_xml_layout(labels)
    except ValueError: return None
    found = place(image, x)
    if found is None or not layout.lines: return None
    placed, stands = found
    boxes = np.array([[line.box[0], line.box[2]] for line in layout.lines]) / layout.width
    def inside(cut: float) -> int: return int(((boxes[:, 0] + 0.002 < cut) & (cut < boxes[:, 1] - 0.002)).sum())
    lefts = boxes[boxes.mean(axis=1) < x]; rights = boxes[boxes.mean(axis=1) >= x]
    gap = float(np.percentile(rights[:, 0], 5) - np.percentile(lefts[:, 1], 95)) if len(lefts) >= 3 and len(rights) >= 3 else None
    return {"image": image, "net": x, "placed": placed, "net_inside": inside(x), "placed_inside": inside(placed), "lines": len(boxes), "gap": gap, "stands": stands}


if __name__ == "__main__":
    pool, count = sys.argv[1], int(sys.argv[2]); out = sys.argv[3] if len(sys.argv) > 3 else None
    found = lance.dataset("runs/htr/curate/gutters.lance").to_table(filter=f"pool = '{pool}' AND gutter_score >= 0.5").to_pandas().sample(count, random_state=1)
    labels = lance.dataset("runs/htr/curate/pages.lance").to_table(columns=["image", "labels"], filter=f"pool = '{pool}'").to_pandas().drop_duplicates("image").set_index("image").labels
    rows = list(zip(found.image, labels.loc[found.image], found.gutter_x))
    with Pool(24) as workers: judged = [r for r in workers.imap_unordered(judge, rows, chunksize=20) if r]
    d = pd.DataFrame(judged)
    if out: d.to_json(out, orient="records", lines=True)
    d["moved"] = (d.placed - d.net).abs()
    print("a fold found in %.2f of scans (stands: quartiles %s)" % ((d.stands >= CLEAR).mean(), d.stands.quantile([.25, .5, .75]).round(1).tolist()))
    print(len(d), "scans; the cut moved by a median of %.4f of the width, over 0.01 in %.2f of scans" % (d.moved.median(), (d.moved > 0.01).mean()))
    def line(name: str, g: pd.DataFrame) -> None:
        print(f"{name:34s} n={len(g):5d}  cut inside >=1 box: network {(g.net_inside >= 1).mean():.3f} placed {(g.placed_inside >= 1).mean():.3f}   >=3 boxes: network {(g.net_inside >= 3).mean():.3f} placed {(g.placed_inside >= 3).mean():.3f}")
    line("all", d); two = d[d.gap.notna()]
    line("lines on both pages", two); line("  gap under 2 % of the width", two[two.gap < 0.02]); line("  gap 2-5 %", two[(two.gap >= 0.02) & (two.gap < 0.05)]); line("  gap over 5 %", two[two.gap >= 0.05])
