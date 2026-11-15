"""INSID3 (visinf/INSID3 at 0c165a1, its own code) segmenting "writing" on our pages from reference pages: a prose
page with true polygons, a ledger page with its (complete) old polygons, or both. Our DINOv3 ViT-L from the HF cache
is given to it through a thin adapter. Scored on held-out prose pages with true polygons; pictures for hard pages. CPU."""
import sys, json, xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NSp
S = Path(sys.argv[1]); sys.path.insert(0, str(S / "third_party/INSID3"))
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel
from models.insid3 import INSID3
Image.MAX_IMAGE_PIXELS = None; torch.set_num_threads(20)
NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
ROOTS = {"svea": ("/mnt/work/haas/deliveries_old/svea-all", "/mnt/scratch/haas/input/svea"),
         "andraarkiv": ("/mnt/work/haas/output/andraarkiv", "/mnt/work/haas/input/andraarkiv")}
GT = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
OUT = S / "insid3"; OUT.mkdir(exist_ok=True)
LISTS = json.load(open(S / "wm_lists.json")); EXTRA = json.load(open(S / "debias_pages.json"))
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

class Encoder(torch.nn.Module):
    """What INSID3 asks of DINOv3: get_intermediate_layers(x, n=1, reshape=True) -> ([B, C, h, w],), the last layer normed."""
    def __init__(self):
        super().__init__(); self.m = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
    def get_intermediate_layers(self, x, n=1, reshape=True):
        h, w = x.shape[-2] // 16, x.shape[-1] // 16
        t = self.m(pixel_values=x).last_hidden_state[:, -h * w:]
        return (t.reshape(x.shape[0], h, w, -1).permute(0, 3, 1, 2),)
model = INSID3(encoder=Encoder(), image_size=1024, device="cpu", mask_refiner="bilinear").eval()

def polygons(path):
    page = ET.parse(path).getroot().find(".//a:Page", NS); out = []
    for line in page.iterfind(".//a:TextLine", NS):
        if not any(s.get("CONTENT") for s in line.iterfind("a:String", NS)): continue
        poly = line.find("a:Shape/a:Polygon", NS); pts = [float(n) for n in poly.get("POINTS").replace(",", " ").split()] if poly is not None else []
        if len(pts) >= 6: out.append(list(zip(pts[0::2], pts[1::2])))
        else:
            x, y, w, h = (float(line.get(k, 0)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT")); out.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return out
def small(img, long=1600):
    k = long / max(img.size); return img.resize((int(img.width * k), int(img.height * k)), Image.BICUBIC)
def mask_image(polys, size, like):
    m = Image.new("L", like.size); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x / size[0] * like.width, y / size[1] * like.height) for x, y in p], fill=255)
    return m
def load(pool, scan, alto=None):
    a, images = ROOTS[pool]; vol = scan.split("_")[0]; img = Image.open(f"{images}/{vol}/{scan}.jpg").convert("RGB"); s = small(img)
    return s, mask_image(polygons(alto or f"{a}/{vol}/{scan}.xml"), img.size, s)

truths = {p.stem: p for p in GT.glob("*/*/*/alto/*.xml")}
cal = [NSp(**r) for r in LISTS["cal"] if r["scan"] in truths]
REFS = {"prose": load("andraarkiv", cal[0].scan, truths[cal[0].scan]), "ledger": load("svea", "C0100744_00189")}
for name, (im, m) in REFS.items():
    c = Image.composite(Image.new("RGB", im.size, (40, 90, 255)), im, m.point(lambda v: int(v * 0.4))); c.resize((c.width // 2, c.height // 2)).save(OUT / f"ref_{name}.jpg", quality=80)
@torch.no_grad()
def segment(target, refs):
    for r in refs: model.set_reference(REFS[r][0], REFS[r][1])
    model.set_target(target); return model.segment().cpu().numpy()
def iou(a, b): return (a & b).sum() / max(1, (a | b).sum())
rows = []
for j, r in enumerate(cal[120:150]):
    t, m = load("andraarkiv", r.scan, truths[r.scan]); y = np.asarray(m) > 0
    rows.append([iou(segment(t, refs), y) for refs in (("prose",), ("ledger",), ("prose", "ledger"))])
    if j % 10 == 9: print("scored", j + 1, np.array(rows).mean(0).round(3), flush=True)
r = np.array(rows); print("held-out %d prose pages, INSID3 writing mask vs true polygons (IoU): prose reference %.3f | ledger reference %.3f | both %.3f" % (len(r), *r.mean(0)), flush=True)
for j, row in enumerate(EXTRA):
    q = NSp(**row); t, old = load(q.pool, q.scan); W, H = t.width // 3, t.height // 3; base = t.resize((W, H)).convert("L").convert("RGB"); tiles = [(t.resize((W, H)), "page")]
    for refs in (("prose",), ("ledger",), ("prose", "ledger")):
        p = Image.fromarray((segment(t, refs) * 255).astype(np.uint8)).resize((W, H))
        c = Image.composite(Image.new("RGB", (W, H), (235, 120, 20)), base, p.point(lambda v: int(v * 0.7))); tiles.append((c, "INSID3, reference: " + " + ".join(refs)))
    canvas = Image.new("RGB", (len(tiles) * (W + 8), H + 44), "white"); dc = ImageDraw.Draw(canvas)
    for i, (im, name) in enumerate(tiles): canvas.paste(im, (i * (W + 8), 44)); dc.text((i * (W + 8) + 3, 24), name, fill="black", font=FONT)
    dc.text((3, 3), f"{q.pool} {q.scan} lines={q.lines}", fill="black", font=FONT); canvas.save(OUT / f"page_{j}.jpg", quality=82); print("panel", j, q.scan, flush=True)
print("done", flush=True)
