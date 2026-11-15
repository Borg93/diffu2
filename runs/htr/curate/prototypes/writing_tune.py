"""Tune the "writing here" layer against GT PAGE polygons. Features: DINOv3-L, TIPSv2-L (resampled to DINOv3's grid)
and both concatenated; page resolutions 1024 and 1536 on the longer side; heads: linear and a one-hidden-layer MLP.
Trained on the train volumes, threshold and choice on val, confirmed on test (volumes disjoint). IoU is measured on a
mask at 1/8 of the page's pixels, so every grid is judged at the same fineness. Results per kind of page."""
import sys, json, time, itertools
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None; torch.manual_seed(0); np.random.seed(0)
D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/htr/curate/writing_tune"); pages = json.load(open(D / "pages.json"))
RES = [1024, 1536]; SAMPLE = 800; MASK = 8
dino = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m", dtype=torch.bfloat16).cuda().eval()
tips = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True).vision_encoder.cuda().eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(3, 1, 1)
def page_image(p):
    with Image.open(p["image"]) as im:
        im.draft("RGB", (4000, 4000)); k = im.width / (p["x1"] if False else im.width) if False else 1.0
        full = im.convert("RGB")
    # scan pixels: the page table's rectangle is in scan pixels at full size; draft may have shrunk the image
    return full
def crop(p):
    with Image.open(p["image"]) as im:
        W = im.width; im.draft("RGB", (4000, 4000)); k = im.width / W
        return im.convert("RGB").crop((round(p["x0"] * k), round(p["y0"] * k), round(p["x1"] * k), round(p["y1"] * k))), k
def truth(p, size, k):
    m = Image.new("L", size); d = ImageDraw.Draw(m)
    for poly in p["polygons"]: d.polygon([(x * k, y * k) for x, y in poly], fill=255)
    return m
@torch.inference_mode()
def features(img, res):
    s = res / max(img.size); w, h = max(16, round(img.width * s / 16) * 16), max(16, round(img.height * s / 16) * 16)
    x = torch.from_numpy(np.asarray(img.resize((w, h), Image.BICUBIC))).cuda().permute(2, 0, 1).float() / 255
    gh, gw = h // 16, w // 16
    fd = dino(pixel_values=((x - MEAN) / STD)[None].to(torch.bfloat16)).last_hidden_state[0, -gh * gw:].float().reshape(gh, gw, -1)
    w14, h14 = max(14, round(img.width * s / 14) * 14), max(14, round(img.height * s / 14) * 14)
    x14 = F.interpolate(x[None], size=(h14, w14), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ft = tips(x14)[2][0].float().reshape(h14 // 14, w14 // 14, -1)
    ft = F.interpolate(ft.permute(2, 0, 1)[None], size=(gh, gw), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    return fd, ft  # both on DINOv3's grid
def grid_target(p, img, k, gh, gw):
    m = truth(p, img.size, k).resize((gw, gh), Image.BOX); return torch.from_numpy(np.asarray(m)).float().cuda() / 255
# pass 1: training patches
train = [p for p in pages if p["role"] == "train"]; X = {r: {"dino": [], "tips": []} for r in RES}; Y = {r: [] for r in RES}; t0 = time.time()
for n, p in enumerate(train):
    try: img, k = crop(p)
    except OSError: continue
    for r in RES:
        fd, ft = features(img, r); gh, gw = fd.shape[:2]; y = grid_target(p, img, k, gh, gw).reshape(-1)
        idx = torch.randperm(len(y), device="cuda")[:SAMPLE]
        X[r]["dino"].append(fd.reshape(-1, fd.shape[-1])[idx].half().cpu()); X[r]["tips"].append(ft.reshape(-1, ft.shape[-1])[idx].half().cpu()); Y[r].append(y[idx].cpu())
    if n % 200 == 199: print("train pages", n + 1, "%.1f s a page" % ((time.time() - t0) / (n + 1)), flush=True)
heads = {}
for r in RES:
    y = torch.cat(Y[r]).cuda()
    for feat in ("dino", "tips", "both"):
        x = torch.cat([torch.cat(X[r]["dino"]), torch.cat(X[r]["tips"])], 1) if feat == "both" else torch.cat(X[r][feat])
        x = x.cuda().float(); mu, sd = x.mean(0), x.std(0) + 1e-6
        for kind in ("linear", "mlp"):
            net = (nn.Linear(x.shape[1], 1) if kind == "linear" else nn.Sequential(nn.Linear(x.shape[1], 256), nn.GELU(), nn.Linear(256, 1))).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            for step in range(int(__import__("os").environ.get("STEPS", 3000))):
                i = torch.randint(0, len(x), (8192,), device="cuda"); loss = F.binary_cross_entropy_with_logits(net((x[i] - mu) / sd).squeeze(1), y[i]); opt.zero_grad(); loss.backward(); opt.step()
            heads[(r, feat, kind)] = (net.eval(), mu, sd); print(f"trained {r} {feat} {kind} loss {loss.item():.3f}", flush=True)
        del x
X = None
# pass 2: val and test pages, every head, IoU at 1/8 of the page's pixels for thresholds 0.3..0.7
TH = [0.3, 0.4, 0.5, 0.6, 0.7]; rows = []
for p in [q for q in pages if q["role"] in ("val", "test")]:
    try: img, k = crop(p)
    except OSError: continue
    tm = np.asarray(truth(p, img.size, k).resize((max(1, img.width // MASK), max(1, img.height // MASK)), Image.BOX)) > 127
    for r in RES:
        fd, ft = features(img, r)
        for (rr, feat, kind), (net, mu, sd) in heads.items():
            if rr != r: continue
            f = torch.cat([fd, ft], -1) if feat == "both" else (fd if feat == "dino" else ft)
            with torch.inference_mode(): prob = torch.sigmoid(net((f - mu) / sd).squeeze(-1))
            pm = F.interpolate(prob[None, None], size=tm.shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            for th in TH:
                a = pm > th; inter = (a & tm).sum(); rows.append({"page": p["page"], "role": p["role"], "res": r, "feat": feat, "head": kind, "th": th, "iou": float(inter / max(1, (a | tm).sum())), "kind": ("table/ledger" if p["table"] or p["digits"] > 0.15 else "prose") + (" colour" if p["colour"] else " greyscale")})
json.dump(rows, open(D / "results.json", "w"))
import pandas as pd
d = pd.DataFrame(rows); pd.set_option("display.width", 220)
v = d[d.role == "val"].groupby(["res", "feat", "head", "th"]).iou.mean().reset_index().sort_values("iou", ascending=False)
print("best on val:\n", v.head(10).to_string(index=False))
best = v.iloc[0]; print("chosen on val:", dict(best))
for kind_, g in d[d.role == "val"].groupby("kind"):
    b = g.groupby(["res", "feat", "head", "th"]).iou.mean().sort_values(ascending=False); print(f"val best for {kind_}: {b.index[0]} {b.iloc[0]:.3f} (n pages {g.page.nunique()})")
t = d[(d.role == "test")]
for cfg_name, q in (("chosen", (best["res"], best["feat"], best["head"], best["th"])),):
    tt = t[(t.res == q[0]) & (t.feat == q[1]) & (t["head"] == q[2]) & (t.th == q[3])]
    print("TEST", cfg_name, q, "IoU %.3f" % tt.iou.mean(), "| by kind", tt.groupby("kind").iou.mean().round(3).to_dict())
for feat in ("dino", "tips", "both"):
    vv = v[v.feat == feat].iloc[0]; tt = t[(t.res == vv.res) & (t.feat == feat) & (t["head"] == vv["head"]) & (t.th == vv.th)]
    print(f"TEST best-on-val for {feat}: res {vv.res} head {vv["head"]} th {vv.th} -> IoU {tt.iou.mean():.3f} | by kind", tt.groupby("kind").iou.mean().round(3).to_dict())
print("done", flush=True)
