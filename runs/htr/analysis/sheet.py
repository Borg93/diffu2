"""Contact sheets: line crops stacked, each with its label (and a note) under it, to look at many lines at once.

    python sheet.py LINES.parquet FLAG_COLUMN OUT.png [N] [SEED]
"""

import sys

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
WIDTH = 1100


def sheet(rows: pd.DataFrame, out: str, note=lambda r: "") -> None:
    tiles = []
    for _, r in rows.iterrows():
        im = Image.open(r.image).convert("RGB")
        scale = min(WIDTH / im.width, 160 / im.height, 3.0)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        hyp = getattr(r, "hyp", None)
        tile = Image.new("RGB", (WIDTH, im.height + (90 if hyp is not None else 62)), "white")
        tile.paste(im, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, im.height + 4), f"GT: {r.text[:95]}", fill=(180, 0, 0), font=FONT)
        draw.text((4, im.height + 32), f"{r.coll[:40]} | {r.id} | {note(r)}"[:110], fill=(0, 0, 160), font=FONT)
        if hyp is not None:
            draw.text((4, im.height + 60), f"MODEL: {hyp[:92]}", fill=(0, 120, 0), font=FONT)
        draw.line([(0, tile.height - 1), (WIDTH, tile.height - 1)], fill=(0, 0, 0), width=2)
        tiles.append(tile)
    page = Image.new("RGB", (WIDTH, sum(t.height for t in tiles)), "white")
    y = 0
    for t in tiles:
        page.paste(t, (0, y))
        y += t.height
    page.save(out)


if __name__ == "__main__":
    lines = pd.read_parquet(sys.argv[1])
    flag, out = sys.argv[2], sys.argv[3]
    n, seed = int(sys.argv[4]) if len(sys.argv) > 4 else 8, int(sys.argv[5]) if len(sys.argv) > 5 else 0
    pick = lines[lines[flag]].sample(n=min(n, int(lines[flag].sum())), random_state=seed)
    sheet(pick, out, note=lambda r: f"bands {r.bands} dens_z {r.dens_z:.1f} ink {r.ink_frac:.3f}")
