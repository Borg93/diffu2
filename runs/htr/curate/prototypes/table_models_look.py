"""A first look at NVIDIA's table models on our table pages (CURATION E45, E46), on the CPU.

Nemotron Page Elements v3 finds the tables on a page; Nemotron Table Structure v1 finds rows, columns and cells in an
image of a table. Each of the 26 test pages the VL model calls a table goes through both: the table model once on
the whole page and once on the largest table the page-elements model finds. The whole-page run is drawn (the
page-elements model's tables red, rows blue, columns orange; cells are counted, not drawn) and both runs are counted.

    CUDA_VISIBLE_DEVICES= uv run python runs/htr/curate/prototypes/table_models_look.py OUT

Both models' own loaders take cuda:0 when a GPU shows, which on this machine is another user's: run it with an empty
CUDA_VISIBLE_DEVICES. The script stops if a GPU is in sight. The weights are read from the clones in
/mnt/scratch/gabriel/models, not fetched into the Hugging Face cache.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

MODELS = Path("/mnt/scratch/gabriel/models")
for repo in ("nemotron-table-structure-v1", "nemotron-page-elements-v3"):
    sys.path.insert(0, str(MODELS / repo))

from nemotron_page_elements_v3.model import YoloXWrapper as PageWrapper  # noqa: E402
from nemotron_page_elements_v3.page_element_v3 import Exp as PageExp  # noqa: E402
from nemotron_page_elements_v3.utils import postprocess_preds_page_element  # noqa: E402
from nemotron_table_structure_v1.model import YoloXWrapper as TableWrapper  # noqa: E402
from nemotron_table_structure_v1.table_structure_v1 import Exp as TableExp  # noqa: E402
from nemotron_table_structure_v1.utils import postprocess_preds_table_structure  # noqa: E402

SURE = 0.3  # a box counted and drawn as found


def load(exp, wrapper, weights: Path):
    config = exp(); model = config.get_model()
    # the checkpoints hold their score as a numpy number: allow that and nothing else through the safe loader
    numbers = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype, type(np.dtype("float64")), type(np.dtype("int64"))]
    with torch.serialization.safe_globals(numbers):
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["model"], strict=True)
    return wrapper(model, config).eval()


def structure(model, image: np.ndarray) -> dict[str, np.ndarray]:
    """Rows, columns and cells of ``image`` as boxes in its pixels, those scored above SURE."""
    with torch.inference_mode():
        found = model(model.preprocess(image), image.shape)[0]
    boxes, labels, scores = postprocess_preds_table_structure(found, SURE, model.labels)
    size = np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
    return {name: boxes[labels == model.labels.index(name)] * size for name in ("row", "column", "cell")}


if __name__ == "__main__":
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES= (cuda:0 is another user's GPU)"
    torch.set_num_threads(8)
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    pages = load(PageExp, PageWrapper, MODELS / "nemotron-page-elements-v3/nemotron_page_elements_v3/weights.pth")
    tables = load(TableExp, TableWrapper, MODELS / "nemotron-table-structure-v1/nemotron_table_structure_v1/weights.pth")
    rows = pd.read_json("runs/htr/curate/e22_rows.json").drop_duplicates("id")
    rows = rows[rows.writing == "table"].assign(cer=lambda d: d.edits / d.chars.clip(lower=1)).sort_values("edits", ascending=False)
    counted = []
    for page in rows.itertuples():
        image = np.asarray(Image.open(f"data/htr/nemotron_parse_pages_pages/{page.id}.jpg").convert("RGB"))
        with torch.inference_mode():
            found = pages(pages.preprocess(image), image.shape)[0]
        boxes, labels, scores = postprocess_preds_page_element(found, pages.config.thresholds_per_class, pages.labels)
        size = np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        kinds = [pages.labels[int(label)] for label in labels]
        held = boxes[[kind == "table" for kind in kinds]] * size if len(boxes) else np.zeros((0, 4))
        whole = structure(tables, image)
        cut = {"row": [], "column": [], "cell": []}
        if len(held):
            x0, y0, x1, y1 = (int(v) for v in held[np.argmax((held[:, 2] - held[:, 0]) * (held[:, 3] - held[:, 1]))])
            cut = structure(tables, np.ascontiguousarray(image[y0:y1, x0:x1]))
        counted.append({"page": page.id, "cer": round(page.cer, 3), "elements": ", ".join(f"{kinds.count(k)} {k}" for k in sorted(set(kinds))),
                        "tables": len(held), "table share": round(float(((held[:, 2] - held[:, 0]) * (held[:, 3] - held[:, 1])).sum() / (image.shape[0] * image.shape[1])), 2) if len(held) else 0.0,
                        "rows": len(whole["row"]), "columns": len(whole["column"]), "cells": len(whole["cell"]),
                        "rows, cut": len(cut["row"]), "columns, cut": len(cut["column"]), "cells, cut": len(cut["cell"])})
        shown = Image.fromarray(image); pen = ImageDraw.Draw(shown); thick = max(2, image.shape[1] // 400)
        for x0, y0, x1, y1 in whole["row"]: pen.rectangle((x0, y0, x1, y1), outline=(40, 110, 220), width=thick)
        for x0, y0, x1, y1 in whole["column"]: pen.rectangle((x0, y0, x1, y1), outline=(240, 140, 20), width=thick)
        for x0, y0, x1, y1 in held: pen.rectangle((x0, y0, x1, y1), outline=(220, 30, 30), width=2 * thick)
        shown.thumbnail((700, 900)); shown.save(out / f"{page.id}.jpg", quality=85)
        print(counted[-1], flush=True)
    table = pd.DataFrame(counted); table.to_csv(out / "counts.csv", index=False)
    for at in range(0, len(table), 9):
        sheet = Image.new("RGB", (3 * 700, 3 * 900), "white")
        for k, name in enumerate(table.page[at:at + 9]):
            shown = Image.open(out / f"{name}.jpg"); sheet.paste(shown, ((k % 3) * 700, (k // 3) * 900))
        sheet.save(out / f"sheet{at // 9}.jpg", quality=85)
    print(table.drop(columns=["page"]).describe().loc[["mean", "50%", "min", "max"]].round(1).to_string())
