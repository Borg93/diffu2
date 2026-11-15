"""Prototype of tests 1 and 2: DINOv3 and TIPSv2 patch features of the same pages. Per backbone: the Space's PCA view
(3 whitened components -> sigmoid -> RGB) and a linear "writing here" probe trained on calibration pages' true
polygons. Then: old polygons vs each writing map, and the two maps against each other."""
import sys, json, xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NSp
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None
NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
ROOTS = {"svea": ("/mnt/work/haas/deliveries_old/svea-all", "/mnt/scratch/haas/input/svea"),
         "andraarkiv": ("/mnt/work/haas/output/andraarkiv", "/mnt/work/haas/input/andraarkiv")}
GT = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LISTS = json.load(open(sys.argv[2]))
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
LONG = 1280
COMMON = (80, 96)  # the grid (rows, columns) both maps are compared on
dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).cuda().eval()
tips = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True).vision_encoder.cuda().eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def sized(img, patch):
    k = LONG / max(img.size)
    return img.resize((max(patch, round(img.width * k / patch) * patch), max(patch, round(img.height * k / patch) * patch)), Image.BICUBIC)

@torch.inference_mode()
def grids(img):
    a = sized(img, 16); x = torch.from_numpy(np.asarray(a)).permute(2, 0, 1).float() / 255
    t = dino(pixel_values=((x - MEAN) / STD)[None].cuda().to(torch.bfloat16)).last_hidden_state[0]
    gd = t[-(a.height // 16) * (a.width // 16):].float().reshape(a.height // 16, a.width // 16, -1)
    b = sized(img, 14); y = torch.from_numpy(np.asarray(b)).permute(2, 0, 1).float() / 255  # TIPS takes 0..1 pixels, as its Space feeds it
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = tips(y[None].cuda())[2][0]
    gt = p.float().reshape(b.height // 14, b.width // 14, -1)
    return {"dino": gd, "tips": gt}

def polygons(path):
    page = ET.parse(path).getroot().find(".//a:Page", NS); out = []
    for line in page.iterfind(".//a:TextLine", NS):
        if not any(s.get("CONTENT") for s in line.iterfind("a:String", NS)): continue
        poly = line.find("a:Shape/a:Polygon", NS)
        pts = [float(n) for n in poly.get("POINTS").replace(",", " ").split()] if poly is not None else []
        if len(pts) >= 6: out.append(list(zip(pts[0::2], pts[1::2])))
        else:
            x, y, w, h = (float(line.get(k, 0)) for k in ("HPOS", "VPOS", "WIDTH", "HEIGHT")); out.append([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return out

def mask(polys, size, hw):
    gh, gw = hw; m = Image.new("L", (gw * 4, gh * 4)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x / size[0] * gw * 4, y / size[1] * gh * 4) for x, y in p], fill=255)
    return torch.from_numpy(np.asarray(m.resize((gw, gh), Image.BOX))).float().cuda() / 255

def pca_rgb(grid):  # the TIPSv2 Space's vis_pca: 3 whitened components, sigmoid(2x)
    f = grid.reshape(-1, grid.shape[-1]); f = f - f.mean(0)
    u, s, v = torch.pca_lowrank(f, q=3, center=False); c = (f @ v[:, :3]) / (s[:3] / (len(f) - 1) ** 0.5)
    return Image.fromarray((torch.sigmoid(2 * c).reshape(*grid.shape[:2], 3).cpu().numpy() * 255).astype(np.uint8))

def common(m): return F.interpolate(m[None, None], size=COMMON, mode="bilinear", align_corners=False)[0, 0]

# --- probes on calibration pages
truths = {p.stem: p for p in GT.glob("*/*/*/alto/*.xml")}
cal = [NSp(**r) for r in LISTS["cal"] if r["scan"] in truths]
X, Y, held = {"dino": [], "tips": []}, {"dino": [], "tips": []}, []
for i, r in enumerate(cal):
    img = Image.open(f"{ROOTS['andraarkiv'][1]}/{r.scan.split('_')[0]}/{r.scan}.jpg").convert("RGB"); g = grids(img); polys = polygons(truths[r.scan])
    if i < 240:
        for k in g: X[k].append(g[k].reshape(-1, g[k].shape[-1])); Y[k].append(mask(polys, img.size, g[k].shape[:2]).reshape(-1))
    else: held.append((g, polys, img.size))
probes = {}
for k in ("dino", "tips"):
    x, y = torch.cat(X[k]), torch.cat(Y[k]); mu, sd = x.mean(0), x.std(0) + 1e-6
    lin = torch.nn.Linear(x.shape[1], 1).cuda(); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(1500):
        idx = torch.randint(0, len(x), (65536,), device="cuda")
        loss = F.binary_cross_entropy_with_logits(lin((x[idx] - mu) / sd).squeeze(1), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    probes[k] = (lin, mu, sd); print(k, "probe: patches", len(x), "writing share %.3f loss %.3f" % (y.mean().item(), loss.item()), flush=True)
    del x, y
X = Y = None
@torch.inference_mode()
def writing(g): return {k: torch.sigmoid(probes[k][0]((g[k] - probes[k][1]) / probes[k][2]).squeeze(-1)) for k in g}
def prf(a, b):
    inter = (a & b).sum().item(); return inter / max(1, a.sum().item()), inter / max(1, b.sum().item()), inter / max(1, (a | b).sum().item())
for k in ("dino", "tips"):
    s = np.array([prf(writing(g)[k] > 0.5, mask(polys, size, g[k].shape[:2]) > 0.5) for g, polys, size in held])
    print(f"{k} held-out 60 pages vs TRUE polygons: map inside polygons %.3f, polygons covered by map %.3f, IoU %.3f" % tuple(s.mean(0)), flush=True)
s = np.array([prf(common(writing(g)["dino"]) > 0.5, common(writing(g)["tips"]) > 0.5)[2] for g, _, _ in held]); print("held-out: IoU between the two backbones' maps %.3f" % s.mean(), flush=True)
pcs = []  # does the first PCA component find writing by itself?
for g, polys, size in held:
    row = []
    for k in g:
        f = g[k].reshape(-1, g[k].shape[-1]); f = f - f.mean(0); _, _, v = torch.pca_lowrank(f, q=3, center=False); y = mask(polys, size, g[k].shape[:2]).reshape(-1)
        row.append(max(abs(np.corrcoef((f @ v[:, j]).cpu().numpy(), y.cpu().numpy())[0, 1]) for j in range(3)))
    pcs.append(row)
print("held-out: best |correlation| of a top-3 PCA component with the true polygon mask: dino %.2f tips %.2f" % tuple(np.array(pcs).mean(0)), flush=True)

def measure(pool, scan):
    alto, images = ROOTS[pool]; vol = scan.split("_")[0]
    img = Image.open(f"{images}/{vol}/{scan}.jpg").convert("RGB"); g = grids(img); w = writing(g); polys = polygons(f"{alto}/{vol}/{scan}.xml")
    out = {}
    for k in g:
        on, cov, _ = prf(mask(polys, img.size, g[k].shape[:2]) > 0.5, w[k] > 0.5)  # polygons on writing; writing covered by polygons
        out[k] = {"polygons_on_writing": on, "writing_covered": cov, "writing_share": (w[k] > 0.5).float().mean().item()}
    out["maps_iou"] = prf(common(w["dino"]) > 0.5, common(w["tips"]) > 0.5)[2]
    return img, g, w, polys, out

def panel(r):
    img, g, w, polys, m = measure(r.pool, r.scan); H = 480; W = int(img.width * H / img.height)
    base = img.resize((W, H)); a = base.copy(); d = ImageDraw.Draw(a)
    for p in polys: d.polygon([(x / img.width * W, y / img.height * H) for x, y in p], outline=(40, 90, 255))
    tiles = [a, pca_rgb(g["dino"]).resize((W, H), Image.NEAREST), pca_rgb(g["tips"]).resize((W, H), Image.NEAREST)]
    for k, colour in (("dino", (235, 120, 20)), ("tips", (20, 150, 150))):
        heat = Image.fromarray((w[k].cpu().numpy() * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
        c = Image.composite(Image.new("RGB", (W, H), colour), base.convert("L").convert("RGB"), heat.point(lambda v: int(v * 0.75))); dc = ImageDraw.Draw(c)
        for p in polys: dc.polygon([(x / img.width * W, y / img.height * H) for x, y in p], outline=(40, 90, 255))
        tiles.append(c)
    canvas = Image.new("RGB", (5 * W + 40, H + 44), "white"); dc = ImageDraw.Draw(canvas)
    for i, (t, name) in enumerate(zip(tiles, ("page + old polygons", "DINOv3 PCA", "TIPSv2 PCA", "DINOv3 writing map", "TIPSv2 writing map"))):
        canvas.paste(t, (i * (W + 10), 44)); dc.text((i * (W + 10) + 3, 24), name, fill="black", font=FONT)
    dc.text((3, 3), f"{r.pool} {r.scan} lines={r.lines} PC={r.page_confidence:.3f} | writing covered by polygons: DINO {m['dino']['writing_covered']:.2f} TIPS {m['tips']['writing_covered']:.2f} | polygons on writing: {m['dino']['polygons_on_writing']:.2f} / {m['tips']['polygons_on_writing']:.2f} | maps IoU {m['maps_iou']:.2f}", fill="black", font=FONT)
    return canvas, m

for name in ("nolines", "few", "random", "many"):
    for i, row in enumerate(LISTS[name]):
        r = NSp(**row)
        try:
            canvas, m = panel(r); canvas.save(OUT / f"{name}_{i}.jpg", quality=80)
            print(name, i, r.scan, "L=%d" % r.lines, {k: round(v, 2) for k, v in m["dino"].items()}, {k: round(v, 2) for k, v in m["tips"].items()}, "iou %.2f" % m["maps_iou"], flush=True)
        except Exception as e: print("failed", r.scan, repr(e)[:200], flush=True)
res = []
for row in LISTS["big"]:
    r = NSp(**row)
    try:
        _, _, _, _, m = measure(r.pool, r.scan); res.append({**row, **{f"{k}_{n}": v for k in ("dino", "tips") for n, v in m[k].items()}, "maps_iou": m["maps_iou"]})
    except Exception as e: print("failed", r.scan, repr(e)[:200], flush=True)
json.dump(res, open(OUT / "sample400.json", "w")); print("big sample done", len(res))
