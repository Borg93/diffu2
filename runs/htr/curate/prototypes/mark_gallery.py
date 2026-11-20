"""A gallery for marking in bulk whether a scan is to be cut, cluster by cluster (CURATION E34).

A plain page served by Python's own http server. The 14,985 scans on the embedding map (scan_map.lance) are grouped
by their cluster; a cluster is picked from a list that says how many scans it holds and what the first form calls
them, and its scans are shown 120 small thumbnails a screen, each starting at the first form's guess. An answer is
picked (keys 1 to 4) and a click on a scan gives it that answer, or one button gives it to every scan on the screen;
frames change in the browser with no round trip, and "Save and next" (Enter) stores the screen. Shift-click opens a
scan large. Marks go to cut_marks.lance (image, mark, guess, marked_at); the last row of an image is its mark.

    uv run python runs/htr/curate/prototypes/mark_gallery.py     # then open http://localhost:7862

UNSURE=<parquet of image, pool, cluster, guess> (not GROUPS: bash keeps that name for itself) shows those scans in those groups instead (unsure_groups.py makes one
of the scans the classifiers are least sure of); NAMES=<json of cluster: name> names the groups in the list.
"""
import os
import json
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
THUMBS = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs_small"); LARGE = Path("/mnt/scratch/gabriel/htr/cut_mark_thumbs"); SCREEN = 120
ROW = pa.schema([("image", pa.utf8()), ("mark", pa.utf8()), ("guess", pa.utf8()), ("marked_at", pa.timestamp("s", "UTC"))])


def thumb(image: str, folder: Path = THUMBS, side: int = 320) -> Path:
    path = folder / f"{sha1(image.encode()).hexdigest()}.jpg"
    if not path.exists():
        with Image.open(image) as scan:
            scan.draft("RGB", (side * 2, side * 2)); scan = scan.convert("RGB"); scan.thumbnail((side, side)); scan.save(path, "JPEG", quality=78)
    return path


def load_scans() -> pd.DataFrame:
    """The map's scans with their cluster and the first form's guess, in the order they are shown: by cluster."""
    placed = lance.dataset(str(HOME / "scan_map.lance")).to_table(columns=["image", "cluster"]).to_pandas()
    said = lance.dataset(str(HOME / "scans.lance")).to_table(columns=["image", "pool", "vl_layout", "vl_table_across_fold"], filter="vl_layout IS NOT NULL").to_pandas()
    scans = placed.merge(said, on="image", how="left")
    scans["guess"] = "cut"
    scans.loc[scans.vl_layout.isin(["not_a_page", "several_loose_sheets"]), "guess"] = "not_a_page"
    scans.loc[scans.vl_layout == "single_page", "guess"] = "single"
    scans.loc[scans.vl_table_across_fold.fillna(False).astype(bool), "guess"] = "no_cut"
    return scans.sort_values(["cluster", "guess", "image"]).reset_index(drop=True)[["image", "pool", "cluster", "guess"]]


sample = pd.read_parquet(os.environ["UNSURE"]).sort_values(["cluster", "guess", "image"]).reset_index(drop=True) if os.environ.get("UNSURE") else load_scans()
NAMES = {int(k): v for k, v in json.load(open(os.environ["NAMES"])).items()} if os.environ.get("NAMES") else {}
for folder in (THUMBS, LARGE): folder.mkdir(parents=True, exist_ok=True)
rows_of = {int(cluster): rows.index.to_list() for cluster, rows in sample.groupby("cluster")}


def held() -> dict[str, str]:
    if not MARKS.exists(): return {}
    rows = lance.dataset(str(MARKS)).to_table().to_pandas().sort_values("marked_at")
    return dict(zip(rows.image, rows.mark))


def clusters() -> list[dict]:
    """Every cluster with its size, how many of its scans are saved and what the first form calls them; those with
    the most scans not guessed "cut" first, since those are the ones that need eyes."""
    saved = held(); listed = []
    for cluster, rows in rows_of.items():
        part = sample.loc[rows]; mix = part.guess.value_counts()
        listed.append({"cluster": cluster, "name": NAMES.get(cluster, ""), "scans": len(rows), "saved": int(part.image.isin(saved.keys()).sum()), "other": int(len(rows) - mix.get("cut", 0)),
                       "mix": ", ".join(f"{name.replace('_', ' ')} {count}" for name, count in mix.items())})
    return sorted(listed, key=lambda row: -row["other"] / row["scans"])


def screen(cluster: int, number: int) -> dict:
    saved = held(); rows = rows_of[cluster]; pages = max(1, -(-len(rows) // SCREEN)); number = min(max(0, number), pages - 1)
    shown = rows[number * SCREEN:(number + 1) * SCREEN]
    return {"cluster": cluster, "screen": number, "screens": pages, "scans": len(rows), "saved": int(sample.loc[rows].image.isin(saved.keys()).sum()),
            "items": [{"at": at, "mark": saved.get(sample.image[at], sample.guess[at]), "guess": sample.guess[at], "saved": sample.image[at] in saved} for at in shown]}


def save(marks: dict[str, str]) -> None:
    now = datetime.now(UTC).replace(microsecond=0); rows = [(int(at), mark) for at, mark in marks.items()]
    table = pa.table({"image": [sample.image[at] for at, _ in rows], "mark": [mark for _, mark in rows], "guess": [sample.guess[at] for at, _ in rows], "marked_at": [now] * len(rows)}, schema=ROW)
    lance.write_dataset(table, str(MARKS), mode="append" if MARKS.exists() else "create")


PAGE = """<!doctype html><meta charset="utf-8"><title>Cut or not</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#16181c;color:#e8e8e8}
header{position:sticky;top:0;background:#16181c;padding:8px 16px;border-bottom:1px solid #333;z-index:5;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
button,select{font:inherit;padding:6px 12px;border-radius:6px;border:2px solid #555;background:#23262b;color:#e8e8e8;cursor:pointer}
button.on{background:#e8e8e8;color:#16181c}
.cut{border-color:#1baf7a}.no_cut{border-color:#eb6834}.single{border-color:#2a78d6}.not_a_page{border-color:#8c8c8c}.odd{border-color:#e87ba4}.print{border-color:#8a5cd0}.illustration{border-color:#c9a227}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px;padding:10px 16px}
#grid div{border:5px solid;border-radius:4px;cursor:pointer;background:#000;position:relative;line-height:0}
#grid span{position:absolute;left:0;bottom:0;right:0;font-size:12px;font-weight:600;padding:2px 6px;line-height:1.5;color:#fff;text-align:center}
#grid .cut span{background:#1baf7a}#grid .no_cut span{background:#eb6834}#grid .single span{background:#2a78d6}#grid .not_a_page span{background:#8c8c8c}#grid .odd span{background:#e87ba4}#grid .print span{background:#8a5cd0}#grid .illustration span{background:#c9a227}
#grid div.sel{outline:5px solid #eda100;outline-offset:1px}#grid div.sel img{opacity:.75}
#grid img{width:100%;height:135px;object-fit:contain}
#big{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:9}#big img{max-width:96%;max-height:96%}
#help{padding:6px 16px;color:#bbb;font-size:13.5px;line-height:1.5;border-bottom:1px solid #333}#help b{color:#e8e8e8}
</style>
<div id="help"><b>1. Click scans to select them</b> (yellow outline), or press <b>Select all</b>. &nbsp; <b>2. Press an answer</b> (or key 1-7): the selected scans get it. &nbsp; <b>3. Save this screen.</b><br>
Every scan shows its current answer under it; at first that is the system's guess. Scans you don't touch keep it. Saving stores every scan on the screen with the answer shown.<br>
<b>Not a page</b>: nothing to read (cover, blank, card). <b>Single page</b>: one page only. <b>Do not cut</b>: two pages with one table or ledger whose rows run across the middle. <b>Cut</b>: two pages, each with its own text. <b>Odd layout</b>: handwriting, but none of the four fits (several loose sheets, a folded-out sheet, pages at angles, writing turned sideways, a small slip on a page). &nbsp; Shift-click: see a scan large.</div>
<header><b style="color:#eda100">Cluster gallery</b><select id="clusters"></select><b id="where"></b>
<button id="all">Select all</button><button id="none">Select none</button><span id="count" style="color:#eda100"></span>
<span id="answers"></span>
<button id="back">&larr; Save, back</button><button id="next" style="border-color:#e8e8e8">Save this screen and next &rarr;</button></header>
<div id="grid"></div><div id="big"><img></div>
<script>
const SAYS={cut:"1 Cut",no_cut:"2 Do not cut",single:"3 Single page",not_a_page:"4 Not a page",odd:"5 Odd layout",print:"6 Print or type (also with handwriting on it)",illustration:"7 Illustration"}, KEYS=Object.keys(SAYS);
let now=null, list=[];
const el=id=>document.getElementById(id);
function count(){const n=now?now.items.filter(it=>it.sel).length:0;el("count").textContent=n?`${n} selected → give them:`:"select scans, then:";}
function answers(){el("answers").innerHTML="";KEYS.forEach(k=>{const b=document.createElement("button");b.textContent=SAYS[k];b.className=k;b.onclick=()=>give(k);el("answers").append(b)})}
function give(k){const picked=now.items.filter(it=>it.sel);if(!picked.length){el("count").textContent="click some scans first (or Select all), then:";return}
  picked.forEach(it=>{it.mark=k;it.sel=false;it.set()});count()}
function draw(){el("where").textContent=`screen ${now.screen+1} of ${now.screens} · ${now.saved} of ${now.scans} saved`;
  el("grid").innerHTML="";now.items.forEach(it=>{const d=document.createElement("div");d.innerHTML=`<img loading="lazy" src="thumb/${it.at}"><span></span>`;it.sel=false;
    it.set=()=>{d.className=it.mark+(it.sel?" sel":"");d.lastChild.textContent=SAYS[it.mark].slice(2).toUpperCase()+(it.mark!==it.guess?" ✎":"")};it.set();
    d.onclick=e=>{if(e.shiftKey){el("big").firstChild.src="large/"+it.at;el("big").style.display="flex";return}it.sel=!it.sel;it.set();count()};el("grid").append(d)});count();window.scrollTo(0,0)}
async function menu(keep){list=await (await fetch("clusters")).json();el("clusters").innerHTML="";list.forEach(c=>{const o=document.createElement("option");o.value=c.cluster;
  o.textContent=`${c.saved>=c.scans?"✓ ":c.saved?"◐ ":""}${c.name||("Cluster "+c.cluster)} · ${c.scans} scans · ${c.mix}`;el("clusters").append(o)});if(keep!==undefined)el("clusters").value=keep}
async function load(cluster,n){now=await (await fetch(`screen/${cluster}/${n}`)).json();el("clusters").value=cluster;draw()}
async function go(by){const marks={};now.items.forEach(it=>marks[it.at]=it.mark);await fetch("save",{method:"POST",body:JSON.stringify(marks)});
  const next=now.screen+by; if(next>=now.screens){await menu();const i=list.findIndex(c=>c.cluster==now.cluster);const after=list.slice(i+1).concat(list.slice(0,i)).find(c=>c.saved<c.scans)||list[i];load(after.cluster,0)}
  else{menu(now.cluster);load(now.cluster,Math.max(0,next))}}
el("next").onclick=()=>go(1);el("back").onclick=()=>go(-1);el("big").onclick=()=>el("big").style.display="none";
el("all").onclick=()=>{now.items.forEach(it=>{it.sel=true;it.set()});count()};el("none").onclick=()=>{now.items.forEach(it=>{it.sel=false;it.set()});count()};
el("clusters").onchange=e=>load(+e.target.value,0);
document.onkeydown=e=>{if(e.target.tagName==="SELECT")return;if(e.key>="1"&&e.key<="7"){give(KEYS[e.key-1])}else if(e.key==="a"){el("all").click()}else if(e.key==="Enter"){go(1)}else if(e.key==="Escape"){el("big").style.display="none";el("none").click()}};
answers();menu().then(()=>{const first=list.find(c=>c.saved<c.scans)||list[0];load(first.cluster,0)});
</script>"""


class Handler(BaseHTTPRequestHandler):
    def send(self, body: bytes, kind: str, cache: bool = False) -> None:
        self.send_response(200); self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (the server's name for it)
        part = self.path.strip("/").split("/")
        try:
            if part == [""]: self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif part[0] == "clusters": self.send(json.dumps(clusters()).encode(), "application/json")
            elif part[0] == "screen": self.send(json.dumps(screen(int(part[1]), int(part[2]))).encode(), "application/json")
            elif part[0] == "thumb": self.send(thumb(sample.image[int(part[1])]).read_bytes(), "image/jpeg", cache=True)
            elif part[0] == "large": self.send(thumb(sample.image[int(part[1])], LARGE, 1400).read_bytes(), "image/jpeg", cache=True)
            else: self.send_error(404)
        except (ValueError, IndexError, KeyError):
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.strip("/") != "save": self.send_error(404); return
        marks = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not all(mark in ("cut", "no_cut", "single", "not_a_page", "odd", "print", "illustration") and 0 <= int(at) < len(sample) for at, mark in marks.items()): self.send_error(400); return
        save(marks); self.send(b"{}", "application/json")

    def log_message(self, format: str, *args: object) -> None: pass  # noqa: A002 (the server's signature)


if __name__ == "__main__":
    print("http://localhost:7862", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 7862), Handler).serve_forever()
