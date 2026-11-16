"""A gallery for marking in bulk whether a scan is to be cut (CURATION E34).

A plain page served by Python's own http server: sixty small thumbnails a screen, every scan starting at the first
form's guess. An answer is picked (keys 1 to 4) and a click on a scan gives it that answer; the frames change in the
browser, with no round trip, and "Save and next" (Enter) stores the screen's marks. A click with Shift held opens the
scan large. Marks go to cut_marks.lance (image, mark, guess, marked_at); the sample, drawn once, is
cut_marks_sample.lance: 1,129 described scans, screens of one guess.

    uv run python runs/htr/curate/prototypes/mark_gallery.py     # then open http://localhost:7862
"""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); SAMPLE = HOME / "cut_marks_sample.lance"; MARKS = HOME / "cut_marks.lance"
THUMBS = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs_small"); LARGE = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs"); SCREEN = 60
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


def thumb(image: str, folder: Path = THUMBS, side: int = 320) -> Path:
    path = folder / f"{sha1(image.encode()).hexdigest()}.jpg"
    if not path.exists():
        with Image.open(image) as scan:
            scan.draft("RGB", (side * 2, side * 2)); scan = scan.convert("RGB"); scan.thumbnail((side, side)); scan.save(path, "JPEG", quality=78)
    return path


if not SAMPLE.exists():
    lance.write_dataset(pa.Table.from_pandas(draw_sample(), preserve_index=False), str(SAMPLE))
sample = lance.dataset(str(SAMPLE)).to_table().to_pandas()
for folder in (THUMBS, LARGE): folder.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(24) as workers: list(workers.map(thumb, sample.image))
screens = [list(range(start, min(start + SCREEN, len(sample)))) for start in range(0, len(sample), SCREEN)]


def held() -> dict[str, str]:
    if not MARKS.exists(): return {}
    rows = lance.dataset(str(MARKS)).to_table().to_pandas().sort_values("marked_at")
    return dict(zip(rows.image, rows.mark))


def screen(number: int) -> dict:
    saved = held(); rows = screens[number]
    return {"screen": number, "screens": len(screens), "saved": len(set(saved) & set(sample.image)), "scans": len(sample), "guess": sample.guess[rows[0]],
            "first_unsaved": next((i for i, r in enumerate(screens) if sample.image[r[0]] not in saved), len(screens) - 1),
            "items": [{"at": at, "mark": saved.get(sample.image[at], sample.guess[at]), "guess": sample.guess[at], "pool": sample.pool[at]} for at in rows]}


def save(marks: dict[str, str]) -> None:
    now = datetime.now(UTC).replace(microsecond=0); rows = [(int(at), mark) for at, mark in marks.items()]
    table = pa.table({"image": [sample.image[at] for at, _ in rows], "mark": [mark for _, mark in rows], "guess": [sample.guess[at] for at, _ in rows], "marked_at": [now] * len(rows)}, schema=ROW)
    lance.write_dataset(table, str(MARKS), mode="append" if MARKS.exists() else "create")


PAGE = """<!doctype html><meta charset="utf-8"><title>Cut or not</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#16181c;color:#e8e8e8}
header{position:sticky;top:0;background:#16181c;padding:10px 16px;border-bottom:1px solid #333;z-index:5;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:7px 14px;border-radius:6px;border:2px solid #555;background:#23262b;color:#e8e8e8;cursor:pointer}
button.on{background:#e8e8e8;color:#16181c}
.cut{border-color:#1baf7a}.no_cut{border-color:#eb6834}.single{border-color:#2a78d6}.not_a_page{border-color:#8c8c8c}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:8px;padding:12px 16px}
#grid div{border:6px solid;border-radius:4px;cursor:pointer;background:#000;position:relative;line-height:0}
#grid div.changed::after{content:"changed";position:absolute;top:4px;left:4px;background:#fff;color:#000;font-size:11px;padding:1px 5px;line-height:1.4;border-radius:3px}
#grid img{width:100%;height:170px;object-fit:contain}
#big{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9}#big img{max-width:96%;max-height:96%}
small{color:#aaa}
</style>
<header><b id="where"></b>
<span id="brushes"></span>
<button id="back">&larr; Save and back</button><button id="next" style="border-color:#e8e8e8">Save and next &rarr; (Enter)</button>
<small>Pick an answer (keys 1-4), click the scans that should have it; click again to undo. Shift-click: see it large.</small></header>
<div id="grid"></div><div id="big"><img></div>
<script>
const SAYS={cut:"1 Cut",no_cut:"2 Do not cut",single:"3 Single page",not_a_page:"4 Not a page"}, KEYS=Object.keys(SAYS);
let brush="cut", now=null;
const el=id=>document.getElementById(id);
function brushes(){el("brushes").innerHTML="";KEYS.forEach(k=>{const b=document.createElement("button");b.textContent=SAYS[k];b.className=k+(k===brush?" on":"");b.onclick=()=>{brush=k;brushes()};el("brushes").append(b)})}
function draw(){el("where").textContent=`Screen ${now.screen+1} of ${now.screens} · ${now.saved} of ${now.scans} saved · these start as: ${SAYS[now.guess].slice(2)}`;
  el("grid").innerHTML="";now.items.forEach(it=>{const d=document.createElement("div");const set=()=>{d.className=it.mark+(it.mark!==it.guess?" changed":"")};set();
    d.innerHTML=`<img loading="lazy" src="thumb/${it.at}">`;
    d.onclick=e=>{if(e.shiftKey){el("big").firstChild.src="large/"+it.at;el("big").style.display="flex";return}it.mark=(it.mark===brush&&it.mark!==it.guess)?it.guess:brush;set()};el("grid").append(d)});window.scrollTo(0,0)}
async function load(n){now=await (await fetch("screen/"+n)).json();draw()}
async function go(by){const marks={};now.items.forEach(it=>marks[it.at]=it.mark);await fetch("save",{method:"POST",body:JSON.stringify(marks)});load(Math.max(0,Math.min(now.screens-1,now.screen+by)))}
el("next").onclick=()=>go(1);el("back").onclick=()=>go(-1);el("big").onclick=()=>el("big").style.display="none";
document.onkeydown=e=>{if(e.key>="1"&&e.key<="4"){brush=KEYS[e.key-1];brushes()}else if(e.key==="Enter"){go(1)}else if(e.key==="Escape"){el("big").style.display="none"}};
brushes();fetch("screen/0").then(r=>r.json()).then(s=>load(s.first_unsaved));
</script>"""


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, kind: str, cache: bool = False) -> None:
        self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
        if cache: self.send_header("Cache-Control", "max-age=86400")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (the server's name for it)
        part = self.path.strip("/").split("/")
        try:
            if part == [""]: self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif part[0] == "screen": self.send(json.dumps(screen(int(part[1]))).encode(), "application/json")
            elif part[0] == "thumb": self.send(thumb(sample.image[int(part[1])]).read_bytes(), "image/jpeg", cache=True)
            elif part[0] == "large": self.send(thumb(sample.image[int(part[1])], LARGE, 1400).read_bytes(), "image/jpeg", cache=True)
            else: self.send_error(404)
        except (ValueError, IndexError, KeyError):
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.strip("/") != "save": self.send_error(404); return
        marks = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not all(mark in ("cut", "no_cut", "single", "not_a_page") and 0 <= int(at) < len(sample) for at, mark in marks.items()): self.send_error(400); return
        save(marks); self.send(b"{}", "application/json")

    def log_message(self, format: str, *args: object) -> None: pass  # noqa: A002 (the server's signature)


if __name__ == "__main__":
    print("http://localhost:7862", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 7862), Handler).serve_forever()
