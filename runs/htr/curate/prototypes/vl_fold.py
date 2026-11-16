"""Can the vision-language model judge or place an opening's cut from a zoomed strip (CURATION E27)?

Asked for the fold's place on the whole scan it answered 0.50 for three scans in four. Here it sees the band around
the fold only, the full height of the scan.

judge: the 40 Svea scans judged blind by eye on 15 Nov (cut/blind_key.json: the gutter network's cut and the
full-resolution placer's, more than 0.4 % of the width apart), the two cuts drawn as dashed lines in orange and blue,
colours shuffled per scan; which lies on the fold? Compared with the eye's 16 decided calls (14 for the placer).

locate: scans whose fold is a clear dark line (the placer's score at least 20; its place the truth), a band of 14 %
of the width set off-centre at random, a ruler from 0 to 100 along its top and bottom; at which number is the fold?
The error is given as a share of the scan's width.

    uv run python runs/htr/curate/prototypes/vl_fold.py judge|locate [THINK]
"""
import base64, io, json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
MODE = sys.argv[1]; THINK = len(sys.argv) > 2; BAND = 0.14
EYE = {**{n: "blue" for n in (1, 4, 5, 6, 8, 20, 21, 24, 31, 34, 38)}, **{n: "orange" for n in (12, 13, 16, 18, 25)}}  # the eye's calls, by the sheet's colours
ORANGE, BLUE = (255, 140, 0), (0, 200, 255)


def strip(image: str, centre: float) -> tuple[Image.Image, float]:
    """The band of the scan around ``centre`` (a share of its width), and the scan's width in px."""
    with Image.open(image) as scan: scan = scan.convert("RGB")
    w = scan.width; return scan.crop((round((centre - BAND / 2) * w), 0, round((centre + BAND / 2) * w), scan.height)), w


def ask(image: Image.Image, text: str) -> str:
    image.thumbnail((1536, 1536)); held = io.BytesIO(); image.save(held, "JPEG", quality=88)
    body = {"model": "nemotron-3.5-super-vl", "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096 if THINK else 150, "chat_template_kwargs": {"enable_thinking": THINK},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(held.getvalue()).decode()}}]}]}
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as answer:
        return json.load(answer)["choices"][0]["message"]["content"] or ""


def judge(row: dict) -> str:
    band, w = strip(row["image"], (row["net"] + row["placed"]) / 2); draw = ImageDraw.Draw(band); left = round(((row["net"] + row["placed"]) / 2 - BAND / 2) * w)
    colour = {"placed": ORANGE if row["orange"] == "placed" else BLUE, "net": ORANGE if row["orange"] == "net" else BLUE}
    for which in ("placed", "net"):
        x = row[which] * w - left
        for y in range(0, band.height, max(40, band.height // 40)): draw.line([(x, y), (x, y + max(10, band.height // 150))], fill=colour[which], width=max(2, w // 900))
    said = ask(band, "This is a narrow vertical strip from a scan of an open bound volume, around the fold (gutter) between its left and right page. "
                     "Two dashed vertical lines are drawn on it, one ORANGE and one BLUE. A cut along the right line would separate the two pages exactly at the fold, "
                     "without leaving a sliver of one page on the other and without cutting through writing. Which line lies on the fold?\n"
                     'Return only JSON: {"line": "orange" | "blue" | "no_difference", "why": "at most 12 words"}')
    found = re.search(r'"line"\s*:\s*"(orange|blue|no_difference)"', said); return found.group(1) if found else "none"


def locate(row: dict) -> float | None:
    rng = random.Random(row["image"]); centre = row["placed"] + rng.uniform(-0.04, 0.04); band, w = strip(row["image"], centre); band.thumbnail((1536, 1536))
    ruled = Image.new("RGB", (band.width, band.height + 80), "white"); ruled.paste(band, (0, 40)); draw = ImageDraw.Draw(ruled); font = ImageFont.load_default(size=18)
    for tick in range(0, 101, 5):
        x = tick / 100 * (band.width - 1)
        for y0 in (0, band.height + 40):
            draw.line([(x, y0 + (22 if y0 == 0 else 0)), (x, y0 + (40 if y0 == 0 else 18))], fill="black", width=2)
            if tick % 10 == 0: draw.text((min(max(x - 10, 0), band.width - 28), y0 + (0 if y0 == 0 else 20)), str(tick), fill="black", font=font)
    said = ask(ruled, "This is a narrow vertical strip from a scan of an open bound volume, around the fold (gutter) between its left and right page. "
                      "A ruler numbered 0 (left edge) to 100 (right edge) runs along the top and the bottom. At which ruler number is the fold, the line where the two pages meet? "
                      "Read it to the nearest whole number; if the fold leans, give its place at mid-height.\n"
                      'Return only JSON: {"fold": <number 0-100>}')
    found = re.search(r'"fold"\s*:\s*([0-9.]+)', said)
    return None if not found else (centre - BAND / 2 + float(found.group(1)) / 100 * BAND) - row["placed"]


if MODE == "judge":
    key = json.load(open("runs/htr/curate/cut/blind_key.json"))
    with ThreadPoolExecutor(8) as pool: said = list(pool.map(judge, key))
    said_which = [("placed" if (s == "orange") == (k["orange"] == "placed") else "net") if s in ("orange", "blue") else s for s, k in zip(said, key)]
    eye = {n: ("placed" if (c == "orange") == (key[n]["orange"] == "placed") else "net") for n, c in EYE.items()}
    print(f"thinking {THINK}: the model's 40 calls: {pd.Series(said_which).value_counts().to_dict()}")
    print(f"  on the eye's 16 decided scans: agrees {sum(said_which[n] == e for n, e in eye.items())}, opposite {sum(said_which[n] in ('placed', 'net') and said_which[n] != e for n, e in eye.items())}, no difference {sum(said_which[n] == 'no_difference' for n in eye)}")
    print(f"  by colour: {pd.Series(said).value_counts().to_dict()}; the scans where it differs from the eye: {[(n, eye[n], said_which[n]) for n in eye if said_which[n] != eye[n]]}")
    print(f"  on the eye's 24 ties: placed {sum(said_which[n] == 'placed' for n in range(40) if n not in eye)}, net {sum(said_which[n] == 'net' for n in range(40) if n not in eye)}, no difference {sum(said_which[n] == 'no_difference' for n in range(40) if n not in eye)}")
else:
    d = pd.read_json("runs/htr/curate/cut/placed_svea_1500.jsonl", lines=True); d = d[d.stands >= 20].sample(120, random_state=4)
    with ThreadPoolExecutor(8) as pool: errors = list(pool.map(locate, d.to_dict("records")))
    e = np.array([x for x in errors if x is not None]); a = np.abs(e)
    print(f"thinking {THINK}: {len(e)} of {len(errors)} answered; error as a share of the scan's width: median {np.median(a):.4f}, 90th percentile {np.quantile(a, 0.9):.4f}; within 0.3 %: {(a <= 0.003).mean():.2f}, within 1 %: {(a <= 0.01).mean():.2f}; bias {e.mean():+.4f}")
    print(f"  for scale: the gutter network on the same scans: median {(d.net - d.placed).abs().median():.4f}, within 0.3 %: {((d.net - d.placed).abs() <= 0.003).mean():.2f}; one ruler step is {BAND / 100:.4f}")
