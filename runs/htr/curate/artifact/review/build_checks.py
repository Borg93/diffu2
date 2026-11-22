"""Samples of the writing layer (CPU) for the review page: the scan, where the layer sees writing (blue), where the old
labels put lines (red outlines). Writes writing.json beside index.html."""
import os
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("OMP_NUM_THREADS", "16")
import base64, io, json, sys
from pathlib import Path
import lance, numpy as np, torch
from PIL import Image, ImageDraw
from diffu_curate.alto import read_alto
from diffu_curate.settings import CurateSettings
from diffu_curate.writing import MASK_STEP, WritingLayer, fitted_size
Image.MAX_IMAGE_PIXELS = None; torch.set_num_threads(16)
out = Path(sys.argv[1]); settings = CurateSettings(); layer = WritingLayer(Path("runs/htr/curate/writing_tune/heads.pt"), device="cpu")
t = lance.dataset("runs/htr/curate/scans.lance").to_table(columns=["image", "pool", "scan", "cut_class", "covered", "filled", "written", "labelled"], filter="pool != 'gt' AND covered IS NOT NULL AND lines > 0").to_pandas()
GROUPS = [("typical", t[(t.cut_class == "cut") & t.covered.between(0.96, 0.99) & t.filled.between(0.92, 0.97)]),
          ("uncovered", t[(t.cut_class == "cut") & (t.covered < 0.7) & (t.written > 0.1)]),
          ("unfilled", t[(t.cut_class == "cut") & (t.filled < 0.6) & (t.labelled > 0.1)]),
          ("table", t[(t.cut_class == "no_cut") & (t.covered < 0.5)])]
result = {}
for name, part in GROUPS:
    items = []
    for r in part.sample(4, random_state=3).itertuples():
        page = read_alto(settings.pool(r.pool).label_file(r.scan))
        with Image.open(r.image) as scan:
            scan.draft("RGB", (2048, 2048)); size = fitted_size(*scan.size); fitted = scan.convert("RGB").resize(size, Image.Resampling.BICUBIC)
        writing = layer.writing(torch.from_numpy(np.asarray(fitted)).permute(2, 0, 1)[None])[0].numpy()
        mask = Image.fromarray((writing * 255).astype(np.uint8)).resize(size, Image.NEAREST)
        shown = Image.composite(Image.blend(fitted, Image.new("RGB", size, (40, 110, 255)), 0.45), fitted, mask)
        pen = ImageDraw.Draw(shown); kx, ky = size[0] / page.width, size[1] / page.height
        for line in page.lines: pen.polygon([(x * kx, y * ky) for x, y in line.outline], outline=(230, 30, 30), width=2)
        buf = io.BytesIO(); shown.save(buf, "JPEG", quality=80)
        items.append({"img": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "note": f"{r.covered:.0%} of the writing under a label · {r.filled:.0%} of the label area holds writing · {r.scan}"})
        print(name, r.scan, round(r.covered, 3), round(r.filled, 3), flush=True)
    result[name] = {"count": len(part), "of": len(t), "items": items}
json.dump(result, open(out, "w")); print(round(out.stat().st_size / 1e6, 1), "MB")
