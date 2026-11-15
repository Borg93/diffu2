"""Does INSID3's positional de-biasing (project DINOv3 features off the subspace a blank image's features span) make
DINOv3's PCA show writing, and does it help the writing probe? Same pages as two_backbones.py. CPU."""
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
LONG, P, KS = 1280, 16, (100, 500)
dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m").eval()
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

@torch.inference_mode()
def patches(x, hw):  # normalised image tensor [3,H,W] -> [h,w,C], each patch L2-normalised as INSID3 does
    t = dino(pixel_values=x[None]).last_hidden_state[0]
    return F.normalize(t[-hw[0] * hw[1]:].reshape(hw[0], hw[1], -1), dim=-1)

# INSID3's positional basis: the features of a blank image (zeros, normalised), centred over positions, SVD
blank = ((torch.zeros(3, LONG, LONG) - MEAN) / STD)
E = patches(blank, (LONG // P, LONG // P)).reshape(-1, 1024).T; E = E - E.mean(1, keepdim=True)
U, Sv, _ = torch.linalg.svd(E, full_matrices=False)
print("blank image: share of its features' variance in the top 10 / 100 / 500 directions: %.3f / %.3f / %.3f" % tuple((Sv[:k] ** 2).sum().item() / (Sv ** 2).sum().item() for k in (10, 100, 500)), flush=True)
PROJ = {k: torch.eye(1024) - U[:, :k] @ U[:, :k].T for k in KS}
def variants(g):  # raw and de-biased grids
    out = {"raw": g}
    for k in KS: out[f"deb{k}"] = F.normalize(g @ PROJ[k], dim=-1)
    return out

def grid(img):
    k = LONG / max(img.size); a = img.resize((max(P, round(img.width * k / P) * P), max(P, round(img.height * k / P) * P)), Image.BICUBIC)
    x = (torch.from_numpy(np.asarray(a)).permute(2, 0, 1).float() / 255 - MEAN) / STD
    return patches(x, (a.height // P, a.width // P))
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
    gh, gw = hw; m = Image.new("L", (gw * 4, gh * 4)); d = ImageDraw.Draw(m)
    for p in polys: d.polygon([(x / size[0] * gw * 4, y / size[1] * gh * 4) for x, y in p], fill=255)
    return torch.from_numpy(np.asarray(m.resize((gw, gh), Image.BOX))).float() / 255
def pca(grid_):
    f = grid_.reshape(-1, grid_.shape[-1]); f = f - f.mean(0); u, s, v = torch.pca_lowrank(f, q=3, center=False)
    return f @ v[:, :3], s[:3] / (len(f) - 1) ** 0.5
def pca_rgb(grid_):
    c, s = pca(grid_); return Image.fromarray((torch.sigmoid(2 * c / s).reshape(*grid_.shape[:2], 3).numpy() * 255).astype(np.uint8))

truths = {p.stem: p for p in GT.glob("*/*/*/alto/*.xml")}
cal = [NSp(**r) for r in LISTS["cal"] if r["scan"] in truths][:160]
names = ["raw"] + [f"deb{k}" for k in KS]
X, Y, held = {n: [] for n in names}, [], []
for i, r in enumerate(cal):
    img = Image.open(f"{ROOTS['andraarkiv'][1]}/{r.scan.split('_')[0]}/{r.scan}.jpg").convert("RGB"); g = grid(img); y = mask(polygons(truths[r.scan]), img.size, g.shape[:2])
    if i < 120:
        for n, v in variants(g).items(): X[n].append(v.reshape(-1, 1024)[::3])
        Y.append(y.reshape(-1)[::3])
    else: held.append((g, y))
    if i % 40 == 39: print("features", i + 1, flush=True)
Y = torch.cat(Y); probes = {}
for n in names:
    x = torch.cat(X[n]); mu, sd = x.mean(0), x.std(0) + 1e-6; lin = torch.nn.Linear(1024, 1); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(600):
        idx = torch.randint(0, len(x), (32768,)); loss = F.binary_cross_entropy_with_logits(lin((x[idx] - mu) / sd).squeeze(1), Y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    probes[n] = (lin, mu, sd)
X = None
@torch.inference_mode()
def writing(v, n): return torch.sigmoid(probes[n][0]((v - probes[n][1]) / probes[n][2]).squeeze(-1))
for n in names:
    iou, corr = [], []
    for g, y in held:
        v = variants(g)[n]; a, b = writing(v, n) > 0.5, y > 0.5; iou.append((a & b).sum().item() / max(1, (a | b).sum().item()))
        c, _ = pca(v); corr.append(max(abs(np.corrcoef(c[:, j].numpy(), y.reshape(-1).numpy())[0, 1]) for j in range(3)))
    print(f"{n}: held-out {len(held)} pages | writing probe IoU vs true polygons %.3f | best top-3 PCA component vs true polygons |corr| %.2f" % (np.mean(iou), np.mean(corr)), flush=True)
for j, row in enumerate(EXTRA):
    r = NSp(**row); alto, images = ROOTS[r.pool]; vol = r.scan.split("_")[0]
    img = Image.open(f"{images}/{vol}/{r.scan}.jpg").convert("RGB"); g = grid(img); v = variants(g); polys = polygons(f"{alto}/{vol}/{r.scan}.xml")
    H = 440; W = int(img.width * H / img.height); base = img.resize((W, H)); a = base.copy(); d = ImageDraw.Draw(a)
    for p in polys: d.polygon([(x / img.width * W, y / img.height * H) for x, y in p], outline=(40, 90, 255))
    tiles = [(a, "page + old polygons")] + [(pca_rgb(v[n]).resize((W, H), Image.NEAREST), "DINOv3 PCA " + {"raw": "as is", "deb100": "position removed (100)", "deb500": "position removed (500)"}[n]) for n in names]
    heat = Image.fromarray((writing(v["deb500"], "deb500").numpy() * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    tiles.append((Image.composite(Image.new("RGB", (W, H), (235, 120, 20)), base.convert("L").convert("RGB"), heat.point(lambda q: int(q * 0.75))), "writing map on de-biased (500)"))
    canvas = Image.new("RGB", (len(tiles) * (W + 8), H + 44), "white"); dc = ImageDraw.Draw(canvas)
    for i, (t, name) in enumerate(tiles): canvas.paste(t, (i * (W + 8), 44)); dc.text((i * (W + 8) + 3, 24), name, fill="black", font=FONT)
    dc.text((3, 3), f"{r.pool} {r.scan} lines={r.lines}", fill="black", font=FONT); canvas.save(OUT / f"page_{j}.jpg", quality=80); print("panel", j, r.scan, flush=True)
print("done", flush=True)
