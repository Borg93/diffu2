"""(writing_eval.py: the same layers at 1,024 px, saved, and judged by what the check is for.)
Tune the "writing here" layer against GT PAGE polygons. Features: DINOv3-L, TIPSv2-L (resampled to DINOv3's grid)
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
RES = [1024]; SAMPLE = 800; MASK = 8
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
torch.save({f"{r}_{feat}_{kind}": (net.state_dict(), mu.cpu(), sd.cpu()) for (r, feat, kind), (net, mu, sd) in heads.items()}, D / "heads.pt")
# pass 2: what the check is for. On val and test pages (labels complete, by the GT):
#  - ceiling: the best IoU any map on the patch grid could reach against the line polygons
#  - region: the map against the polygons closed over the gaps between lines (recall of the polygons, precision in the closed region)
#  - defects: a run of consecutive lines' polygons removed on purpose (10, 25, 50 % of the lines); does "writing covered" fall?
import random
def closed(mask, k=9):
    t = torch.from_numpy(mask)[None, None].float(); t = F.max_pool2d(t, k, 1, k // 2); t = -F.max_pool2d(-t, k, 1, k // 2); return t[0, 0].numpy() > 0.5
def poly_mask(polys, size, k, shape):
    m = Image.new("L", size); d = ImageDraw.Draw(m)
    for poly in polys: d.polygon([(x * k, y * k) for x, y in poly], fill=255)
    return np.asarray(m.resize((shape[1], shape[0]), Image.BOX)) > 127
rows = []
for p in [q for q in pages if q["role"] in ("val", "test")]:
    try: img, k = crop(p)
    except OSError: continue
    shape = (max(1, img.height // MASK), max(1, img.width // MASK)); tm = poly_mask(p["polygons"], img.size, k, shape); region = closed(tm)
    fd, ft = features(img, 1024); gh, gw = fd.shape[:2]
    grid = np.asarray(Image.fromarray((tm * 255).astype(np.uint8)).resize((gw, gh), Image.BOX)) / 255
    up = np.asarray(Image.fromarray((grid * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BILINEAR)) / 255
    ceiling = max(((up > th) & tm).sum() / max(1, ((up > th) | tm).sum()) for th in (0.3, 0.4, 0.5, 0.6))
    rng = random.Random(p["page"]); n = len(p["polygons"]); kept = {}
    for share in (0.10, 0.25, 0.50):
        run = max(1, round(n * share)); start = rng.randrange(0, n - run + 1) if n > run else 0
        keep = p["polygons"][:start] + p["polygons"][start + run:]; km = poly_mask(keep, img.size, k, shape); kept[share] = (km, km.sum() / max(1, tm.sum()))
    kind_ = ("table/ledger" if p["table"] or p["digits"] > 0.15 else "prose") + (" colour" if p["colour"] else " greyscale")
    for (rr, feat, kind), (net, mu, sd) in heads.items():
        f = torch.cat([fd, ft], -1) if feat == "both" else (fd if feat == "dino" else ft)
        with torch.inference_mode(): prob = torch.sigmoid(net((f - mu) / sd).squeeze(-1))
        pm = F.interpolate(prob[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        for th in (0.4, 0.5):
            a = pm > th; w = max(1, a.sum())
            row = {"page": p["page"], "role": p["role"], "kind": kind_, "feat": feat, "head": kind, "th": th, "lines": n, "ceiling": float(ceiling),
                   "iou": float((a & tm).sum() / max(1, (a | tm).sum())), "recall": float((a & tm).sum() / max(1, tm.sum())), "precision_region": float((a & region).sum() / w),
                   "covered_intact": float((a & region).sum() / w)}
            for share, (km, true_share) in kept.items():
                row[f"covered_{int(share * 100)}"] = float((a & closed(km)).sum() / w); row[f"true_{int(share * 100)}"] = float(true_share)
            rows.append(row)
json.dump(rows, open(D / "eval.json", "w"))
import pandas as pd
d = pd.DataFrame(rows); pd.set_option("display.width", 220)
def auc(neg, pos):  # how often a degraded page scores below an intact one
    neg, pos = np.asarray(neg), np.asarray(pos); return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
print("ceiling (best IoU a patch-grid map can reach):", d[d.role == "test"].drop_duplicates("page").groupby("kind").ceiling.mean().round(3).to_dict(), "all %.3f" % d[d.role == "test"].drop_duplicates("page").ceiling.mean())
for role in ("val", "test"):
    out = []
    for (feat, head, th), g in d[d.role == role].groupby(["feat", "head", "th"]):
        r = {"feat": feat, "head": head, "th": th, "iou": g.iou.mean(), "recall": g.recall.mean(), "precision_region": g.precision_region.mean()}
        for s_ in (10, 25, 50): r[f"auc_{s_}"] = auc(g[f"covered_{s_}"], g.covered_intact)
        r["spearman_25"] = pd.concat([g.covered_25, g.covered_intact]).rank().corr(pd.concat([g.true_25, pd.Series(1.0, index=g.index)]).rank())
        out.append(r)
    o = pd.DataFrame(out).round(3).sort_values("auc_25", ascending=False); print(role.upper(), "\n", o.to_string(index=False))
t = d[(d.role == "test") & (d["head"] == "linear") & (d.th == 0.5)]
for feat in ("dino", "tips", "both"):
    g = t[t.feat == feat]; print("TEST", feat, "linear 0.5 by kind:", {k_: {"auc25": round(auc(x.covered_25, x.covered_intact), 3), "auc10": round(auc(x.covered_10, x.covered_intact), 3), "intact_median": round(x.covered_intact.median(), 3), "n": len(x)} for k_, x in g.groupby("kind")})
print("done", flush=True)
