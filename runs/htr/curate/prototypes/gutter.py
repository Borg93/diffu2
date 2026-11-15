"""Prototype: find an opening's gutter from the image (TIPSv2 features pooled per column, a linear layer per column),
trained on GT scans whose two pages both hold text (gutter = middle of the gap between their text blocks) and on
single sheets (no gutter). Then compared with the page builder's cut on the suspect groups."""
import sys, json, random, collections, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel
Image.MAX_IMAGE_PIXELS = None
NS = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
GT = Path.home() / "Desktop/data/GT-Lejonet_HTR_202602"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
random.seed(0)
M, R = [], []
for sp in ("train", "dev", "test"):
    M += [json.loads(l) for l in open(f"data/htr/nemotron_parse_pages.{sp}.meta.jsonl")]; R += open(f"data/htr/nemotron_parse_pages.{sp}.jsonl").read().splitlines()
key = lambda m: (m["collection"], m["volume"], m["scan"])
by = collections.defaultdict(dict)
for i, m in enumerate(M): by[key(m)][m["side"]] = i
altos = {}
for p in GT.glob("*/*/*/alto/*.xml"):
    if not p.name.startswith("._"): altos[(p.parts[-5], p.parts[-4], p.stem)] = p
def paths(k):
    a = altos.get((k[0], k[1], k[2].rsplit(".", 1)[0]))
    if a is None: return None
    scan = a.parent.parent / k[2]
    return (scan, a) if scan.exists() else None
def crop_aspect(i):
    with Image.open(json.loads(R[i])["images"][0]) as im: return im.width / im.height
def lines(alto):
    page = ET.parse(alto).getroot().find(".//a:Page", NS); out = []
    for l in page.iterfind(".//a:TextLine", NS):
        if any(s.get("CONTENT") for s in l.iterfind("a:String", NS)):
            x, w = float(l.get("HPOS", 0)), float(l.get("WIDTH", 0)); out.append((x, x + w))
    return out, float(page.get("WIDTH", 0))

both = [k for k, v in by.items() if "left" in v and "right" in v and paths(k)]
whole = [k for k, v in by.items() if None in v and paths(k)]
single = [k for k, v in by.items() if len(v) == 1 and None not in v and paths(k)]
random.shuffle(both); random.shuffle(whole); random.shuffle(single)
print("scans: both sides", len(both), "whole", len(whole), "one side", len(single), flush=True)

tips = AutoModel.from_pretrained("google/tipsv2-l14", trust_remote_code=True).vision_encoder.cuda().eval()
LONG = 1260
@torch.inference_mode()
def columns(scan):
    img = Image.open(scan).convert("RGB"); k = LONG / max(img.size)
    b = img.resize((max(14, round(img.width * k / 14) * 14), max(14, round(img.height * k / 14) * 14)), Image.BICUBIC)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = tips((torch.from_numpy(np.asarray(b)).permute(2, 0, 1).float() / 255)[None].cuda())[2][0]
    g = p.float().reshape(b.height // 14, b.width // 14, -1)
    return torch.cat([g.mean(0), g.amax(0)], -1), img  # [columns, 2048]

def builder_edge(k):  # the builder's cut as a fraction of the scan's width
    v = by[k]; scan, _ = paths(k)
    with Image.open(scan) as im: aspect = im.width / im.height
    if "left" in v: return crop_aspect(v["left"]) / aspect
    return 1 - crop_aspect(v["right"]) / aspect

def gap(k):  # the gap between the two pages' text, as fractions of the width, around the builder's cut
    scan, alto = paths(k); ls, W = lines(alto); e = builder_edge(k) * W
    left = [b for a, b in ls if (a + b) / 2 < e]; right = [a for a, b in ls if (a + b) / 2 >= e]
    if not left or not right or W <= 0: return None
    lo, hi = float(np.percentile(left, 95)) / W, float(np.percentile(right, 5)) / W
    return (lo, hi) if hi > lo else None

train, test = [], []
for k in both[:900]:
    g = gap(k)
    if g: (train if len(train) < 600 else test).append((k, g))
tall = []
for k in whole:
    scan, _ = paths(k)
    with Image.open(scan) as im:
        if im.width / im.height < 0.78: tall.append(k)
    if len(tall) >= 420: break
neg_train, neg_test = tall[:300], tall[300:420]
print("train openings", len(train), "test openings", len(test), "single sheets", len(neg_train), len(neg_test), flush=True)

X, Y = [], []
for k, g in train:
    f, _ = columns(paths(k)[0]); c = (g[0] + g[1]) / 2 * len(f); y = torch.zeros(len(f)); y[max(0, int(c - 1.5)):int(c + 1.5) + 1] = 1; X.append(f); Y.append(y)
for k in neg_train:
    f, _ = columns(paths(k)[0]); X.append(f); Y.append(torch.zeros(len(f)))
x, y = torch.cat(X), torch.cat(Y).cuda(); mu, sd = x.mean(0), x.std(0) + 1e-6
lin = torch.nn.Linear(x.shape[1], 1).cuda(); opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-3)
pw = torch.tensor((1 - y.mean()) / y.mean()).clamp(max=30.0)
for _ in range(2000):
    loss = F.binary_cross_entropy_with_logits(lin((x - mu) / sd).squeeze(1), y, pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step()
print("gutter layer trained: columns", len(x), "loss %.3f" % loss.item(), flush=True)
@torch.inference_mode()
def find(scan):
    f, img = columns(scan); p = torch.sigmoid(lin((f - mu) / sd).squeeze(1)); p = F.avg_pool1d(p[None, None], 3, 1, 1)[0, 0]
    lo, hi = int(0.15 * len(p)), int(0.85 * len(p)); j = lo + int(p[lo:hi].argmax()); return float(p[j]), (j + 0.5) / len(p), img

res = [(find(paths(k)[0])[:2], g) for k, g in test]
score = np.array([r[0][0] for r in res]); pos = np.array([r[0][1] for r in res]); g = np.array([r[1] for r in res])
inside = (pos >= g[:, 0] - 0.01) & (pos <= g[:, 1] + 0.01)
print("held-out openings: %d; gutter found inside the gap between the text blocks: %.1f%%; median score %.2f; score<0.5: %.1f%%" % (len(res), 100 * inside.mean(), np.median(score), 100 * (score < 0.5).mean()), flush=True)
print("  distance from gap centre (share of width): median %.3f, p90 %.3f" % (np.median(abs(pos - g.mean(1))), np.percentile(abs(pos - g.mean(1)), 90)), flush=True)
neg = np.array([find(paths(k)[0])[0] for k in neg_test]); print("held-out single sheets: %d; gutter score >0.5 (false alarm): %.1f%%; median score %.2f" % (len(neg), 100 * (neg > 0.5).mean(), np.median(neg)), flush=True)
TH = 0.5
def tile(k, title, size=300):
    s, p, img = find(paths(k)[0]); kk = size / img.height; im = img.resize((max(1, int(img.width * kk)), size)); d = ImageDraw.Draw(im)
    v = by[k]
    if None not in v: e = builder_edge(k) * im.width; d.line([(e, 0), (e, size)], fill=(40, 90, 255), width=3)
    if s > TH: d.line([(p * im.width, 0), (p * im.width, size)], fill=(230, 30, 30), width=2)
    return im, f"{title} s={s:.2f}", s, p
def sheet(items, out, cols=6):
    tiles = [tile(k, t) for k, t in items]; w = max(t[0].width for t in tiles) + 8; rows = (len(tiles) + cols - 1) // cols
    c = Image.new("RGB", (cols * w, rows * 322), "white"); d = ImageDraw.Draw(c)
    for j, (im, t, _, _) in enumerate(tiles): c.paste(im, ((j % cols) * w, (j // cols) * 322 + 20)); d.text(((j % cols) * w + 2, (j // cols) * 322 + 3), t[:44], fill="black", font=FONT)
    c.save(out, quality=80)
# suspects 1: whole scans that are nearly square
near = []
for k in whole:
    with Image.open(paths(k)[0]) as im:
        if im.width / im.height >= 0.85: near.append(k)
r1 = [(k, find(paths(k)[0])[:2]) for k in near]; flagged = [k for k, (s, p) in r1 if s > TH]
print("whole scans nearly square (aspect>=0.85): %d; image shows a gutter (score>%.1f): %d" % (len(near), TH, len(flagged)), flush=True)
sheet([(k, k[0][3:20]) for k in flagged[:12]], OUT / "whole_flagged.jpg"); sheet([(k, k[0][3:20]) for k in [k for k in near if k not in flagged][:12]], OUT / "whole_not_flagged.jpg")
# all other whole scans (tall): how many show a gutter?
rest = [k for k in whole if k not in near][:600]; r_rest = [find(paths(k)[0])[0] for k in rest]; print("tall whole scans sampled: %d; gutter score>%.1f: %d" % (len(rest), TH, sum(s > TH for s in r_rest)), flush=True)
# suspects 2: one-side pages: builder's cut vs the image's gutter
r2 = []
for k in single[:400]:
    s, p, _ = find(paths(k)[0]); r2.append((k, s, p, builder_edge(k)))
found = [r for r in r2 if r[1] > TH]; diff = np.array([abs(r[2] - r[3]) for r in found])
print("one-side pages sampled: %d; gutter found in image: %d; builder's cut off by >2%% of width: %d (%.0f%%), >5%%: %d, >10%%: %d; median %.3f" % (len(r2), len(found), (diff > .02).sum(), 100 * (diff > .02).mean(), (diff > .05).sum(), (diff > .1).sum(), np.median(diff)), flush=True)
worst = sorted(found, key=lambda r: -abs(r[2] - r[3])); sheet([(r[0], f"{r[0][0][3:16]} d={abs(r[2]-r[3]):.2f}") for r in worst[:12]], OUT / "single_worst.jpg")
sheet([(r[0], f"{r[0][0][3:16]} d={abs(r[2]-r[3]):.2f}") for r in random.sample(found, 12)], OUT / "single_random.jpg")
sheet([(k, k[0][3:20]) for k, _ in test[:12]], OUT / "both_test.jpg")
# both-side scans: builder's cut vs image
r3 = []
for k, g_ in test:
    s, p, _ = find(paths(k)[0]); r3.append(abs(p - builder_edge(k)))
r3 = np.array(r3); print("both-side scans (held-out): builder's cut vs image gutter: median %.3f; off by >2%%: %.0f%%, >5%%: %.0f%%" % (np.median(r3), 100 * (r3 > .02).mean(), 100 * (r3 > .05).mean()), flush=True)
json.dump({"near": [[list(k), s, p] for k, (s, p) in r1], "single": [[list(r[0]), r[1], r[2], r[3]] for r in r2]}, open(OUT / "results.json", "w"))
print("done", flush=True)
