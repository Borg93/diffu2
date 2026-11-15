"""Does AnyUp (4x finer DINOv3 features, guided by the image) give a sharper "writing here" map than plain bilinear
interpolation of the same features? Same linear probe on both, scored against true line polygons at the fine grid. CPU."""
import sys, json, xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NSp
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None; torch.set_num_threads(20)
NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
ROOTS = {"svea": ("/mnt/work/haas/deliveries_old/svea-all", "/mnt/scratch/haas/input/svea"),
         "andraarkiv": ("/mnt/work/haas/output/andraarkiv", "/mnt/work/haas/input/andraarkiv")}
GT = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LISTS = json.load(open(sys.argv[2])); EXTRA = json.load(open(sys.argv[3]))
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
LONG, P, UP = 1280, 16, 4
dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
anyup = torch.hub.load("wimmerth/anyup", "anyup_multi_backbone", trust_repo=True).eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

@torch.inference_mode()
def low(img):
    k = LONG / max(img.size); a = img.resize((max(P, round(img.width * k / P) * P), max(P, round(img.height * k / P) * P)), Image.BICUBIC)
    x = (torch.from_numpy(np.asarray(a)).permute(2, 0, 1).float() / 255 - MEAN) / STD
    t = dino(pixel_values=x[None]).last_hidden_state[0]; h, w = a.height // P, a.width // P
    return x, t[-h * w:].reshape(h, w, -1)
@torch.inference_mode()
def fine(x, g):  # two finer grids of the same features: AnyUp's and bilinear
    h, w = g.shape[0] * UP, g.shape[1] * UP; f = g.permute(2, 0, 1)[None]
    return anyup(x[None], f, output_size=(h, w), q_chunk_size=64)[0].permute(1, 2, 0), F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
def polygons(path):
    page = ET.parse(path).getroot().find(".//a:Page", NS); out = []
    for line in page.iterfind(".//a:TextLine", NS):
        if not any(s.get("CONTENT") for s in line.iterfind("a:String", NS)): continue
        poly = line.find("a:Shape/a:Polygon", NS); pts = [float(n) for n in poly.get("POINTS").replace(",", " ").split()] if poly is not None else []
        if len(pts) >= 6: out.append(list(zip(pts[0::2], pts[1::2])))
        else:
            x, y, w, h = (float(line.get(k, 0)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT")); out.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return out
def mask(polys, size, hw):
    gh, gw = hw; m = Image.new("L", (gw * 2, gh * 2)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x / size[0] * gw * 2, y / size[1] * gh * 2) for x, y in p], fill=255)
    return torch.from_numpy(np.asarray(m.resize((gw, gh), Image.BOX))).float() / 255

truths = {p.stem: p for p in GT.glob("*/*/*/alto/*.xml")}
cal = [NSp(**r) for r in LISTS["cal"] if r["scan"] in truths][:150]
X, Y, held = [], [], []
for i, r in enumerate(cal):
    img = Image.open(f"{ROOTS['andraarkiv'][1]}/{r.scan.split('_')[0]}/{r.scan}.jpg").convert("RGB"); x, g = low(img); polys = polygons(truths[r.scan])
    if i < 120: X.append(g.reshape(-1, 1024)[::3]); Y.append(mask(polys, img.size, g.shape[:2]).reshape(-1)[::3])
    else: held.append((x, g, polys, img.size))
    if i % 40 == 39: print("features", i + 1, flush=True)
X, Y = torch.cat(X), torch.cat(Y); mu, sd = X.mean(0), X.std(0) + 1e-6; lin = torch.nn.Linear(1024, 1); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
for _ in range(600):
    idx = torch.randint(0, len(X), (32768,)); loss = F.binary_cross_entropy_with_logits(lin((X[idx] - mu) / sd).squeeze(1), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
@torch.inference_mode()
def writing(v): return torch.sigmoid(lin((v - mu) / sd).squeeze(-1))
def iou(a, b): return (a & b).sum().item() / max(1, (a | b).sum().item())
rows = []
for j, (x, g, polys, size) in enumerate(held):
    a, b = fine(x, g); y = mask(polys, size, a.shape[:2]) > 0.5
    rows.append((iou(writing(a) > 0.5, y), iou(writing(b) > 0.5, y), iou(F.interpolate((writing(g))[None, None], size=y.shape, mode="nearest")[0, 0] > 0.5, y)))
    if j % 10 == 9: print("scored", j + 1, np.array(rows).mean(0).round(3), flush=True)
r = np.array(rows); print("held-out %d pages, writing map vs true line polygons at the 4x finer grid: AnyUp %.3f | bilinear %.3f | coarse patches %.3f | pages where AnyUp beats bilinear: %d" % (len(r), *r.mean(0), (r[:, 0] > r[:, 1]).sum()), flush=True)
for j, row in enumerate(EXTRA):
    q = NSp(**row); alto, images = ROOTS[q.pool]; vol = q.scan.split("_")[0]
    img = Image.open(f"{images}/{vol}/{q.scan}.jpg").convert("RGB"); x, g = low(img); a, b = fine(x, g)
    H = 520; W = int(img.width * H / img.height); base = img.resize((W, H)).convert("L").convert("RGB"); tiles = [(img.resize((W, H)), "page")]
    for v, name in ((g, "coarse patches"), (b, "bilinear 4x"), (a, "AnyUp 4x")):
        heat = Image.fromarray((writing(v).numpy() * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR if v is not g else Image.NEAREST)
        tiles.append((Image.composite(Image.new("RGB", (W, H), (235, 120, 20)), base, heat.point(lambda z: int(z * 0.8))), "writing map: " + name))
    canvas = Image.new("RGB", (len(tiles) * (W + 8), H + 44), "white"); dc = ImageDraw.Draw(canvas)
    for i, (t, name) in enumerate(tiles): canvas.paste(t, (i * (W + 8), 44)); dc.text((i * (W + 8) + 3, 24), name, fill="black", font=FONT)
    dc.text((3, 3), f"{q.pool} {q.scan} lines={q.lines}", fill="black", font=FONT); canvas.save(OUT / f"page_{j}.jpg", quality=82); print("panel", j, q.scan, flush=True)
print("done", flush=True)
