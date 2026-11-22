"""PaddleOCR's text detector, untrained on our material, against line outlines (CURATION E69, step 1).

    CUDA_VISIBLE_DEVICES=0 /mnt/scratch/gabriel/envs/paddle/bin/python runs/htr/curate/prototypes/ppocr_det_try.py \
        ITEMS.json MODEL_DIR OUT_DIR LONG_SIDE

ITEMS.json: scans with their image, the label file's page size and line outlines. The detector's ONNX copy runs through
ONNX Runtime (Paddle's own package servers are out of reach from here); the DB post-processing is written out below
with the model's own thresholds (inference.yml). Written: OUT_DIR/pred.json (the found outlines, in scan pixels) and
the numbers printed, for the ground-truth items only: the share of the truth's line area found, the share of the found
area that is truth, the share of the truth's lines found (half their area), and found outlines that span two lines.
"""
import json, sys
from pathlib import Path

import cv2, numpy as np, onnxruntime as ort, pyclipper

THRESH, BOX_THRESH, UNCLIP, MIN_SIDE = 0.2, 0.45, 1.4, 4
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def found(session, image: np.ndarray, long_side: int) -> list[np.ndarray]:
    h, w = image.shape[:2]; k = long_side / max(h, w)
    nw, nh = max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)
    x = ((cv2.resize(image, (nw, nh)).astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)[None]
    prob = session.run(None, {session.get_inputs()[0].name: x})[0][0, 0]
    contours, _ = cv2.findContours((prob > THRESH).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours[:3000]:
        if len(c) < 4: continue
        mask = np.zeros(prob.shape, np.uint8); cv2.fillPoly(mask, [c], 1)
        if cv2.mean(prob, mask)[0] < BOX_THRESH: continue
        c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
        area, length = cv2.contourArea(c), cv2.arcLength(c, True)
        if len(c) < 4 or length == 0: continue
        offset = pyclipper.PyclipperOffset(); offset.AddPath(c.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        grown = offset.Execute(area * UNCLIP / length)
        if len(grown) != 1: continue
        p = np.array(grown[0], np.float32)
        if min(np.ptp(p[:, 0]), np.ptp(p[:, 1])) < MIN_SIDE: continue
        polys.append(p * [w / nw, h / nh])
    return polys


if __name__ == "__main__":
    items, model, out, long_side = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(model / "inference.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("runs on", session.get_providers()[0], flush=True)
    preds, rows = {}, []
    for n, item in enumerate(items):
        image = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        if image is None: continue
        h, w = image.shape[:2]; polys = found(session, image, long_side); preds[item["scan"]] = [p.round(1).tolist() for p in polys]
        if item["kind"] != "gt": continue
        k = 1000 / max(h, w); size = (round(h * k), round(w * k)); kx, ky = w / item["width"] * k, h / item["height"] * k
        truth = [np.round(np.array(l) * [kx, ky]).astype(np.int32) for l in item["lines"]]
        tmask, pmask = np.zeros(size, np.uint8), np.zeros(size, np.uint8)
        cv2.fillPoly(tmask, truth, 1); cv2.fillPoly(pmask, [np.round(p * k).astype(np.int32) for p in polys], 1)
        both = int((tmask & pmask).sum()); hit = 0
        index = np.zeros(size, np.int32)
        for j, t in enumerate(truth, 1): cv2.fillPoly(index, [t], j)
        for j, t in enumerate(truth, 1):
            one = index == j; hit += one.sum() > 0 and (pmask[one].mean() >= 0.5)
        spans = 0
        for p in polys:
            one = np.zeros(size, np.uint8); cv2.fillPoly(one, [np.round(p * k).astype(np.int32)], 1)
            ids, counts = np.unique(index[one > 0], return_counts=True)
            spans += sum(c >= 0.3 * (index == i).sum() for i, c in zip(ids, counts) if i > 0) >= 2
        rows.append((both / max(1, tmask.sum()), both / max(1, pmask.sum()), hit / max(1, len(truth)), len(polys), len(truth), spans))
        if n % 40 == 0: print(n, "of", len(items), flush=True)
    json.dump(preds, open(out / "pred.json", "w"))
    r = np.array(rows, float)
    print(f"{len(r)} ground-truth scans, long side {long_side}: truth's line area found {r[:, 0].mean():.1%} (median {np.median(r[:, 0]):.1%}); found area that is truth {r[:, 1].mean():.1%}; "
          f"truth's lines found {r[:, 2].mean():.1%} (median {np.median(r[:, 2]):.1%}); outlines found {int(r[:, 3].sum())} for {int(r[:, 4].sum())} true lines; outlines spanning two or more lines {int(r[:, 5].sum())}")
