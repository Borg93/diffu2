"""Every pool scan scored by the quality predictor E68 chose (KumoTabular on the scan features, no image vector).

    CUDA_VISIBLE_DEVICES=0 /mnt/scratch/gabriel/envs/sdm/bin/python runs/htr/curate/prototypes/quality_score.py OUT_DIR [LIMIT]

The 3,651 calibration scans are the context; the pool's scans are scored a batch at a time: ``quality_bad`` (the chance
the old label is bad: over a tenth of its matched characters wrong, or of the true lines missed, or of its lines
astray) and ``quality_cer`` (the matched lines' CER it expects). Parquet parts in OUT_DIR, a scan a part holds is
skipped. The calibration scans themselves get the prediction made with their own volume held out (predictions.parquet).
"""
import os, sys, time, warnings

warnings.filterwarnings("ignore"); os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0"); os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
from pathlib import Path

import lance, pandas as pd, sdm, torch

NUM = ["width", "height", "page_confidence", "regions", "lines", "empty_lines", "chars", "words", "digit_share", "line_height_median", "line_height_spread", "line_width_median", "chars_per_100px_median", "odd_density_lines", "polygon_cover", "repeated_lines", "longest_word_run", "compression", "gutter_score", "cut_p", "table_p", "kind_p", "layout_p", "saturation", "language_p", "written", "labelled", "covered", "filled", "colour"]
CAT = ["pool", "cut_class", "kind", "layout", "language", "pool_cluster"]
BATCH = 50_000


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy(); frame["colour"] = frame.colour.astype("float")
    for c in CAT: frame[c] = frame[c].astype("string").fillna("none").astype(str)
    return frame


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    context = tidy(pd.read_parquet("/mnt/scratch/gabriel/htr/quality/calibration.parquet")); n = len(context)
    done = {i for part in out.glob("part-*.parquet") for i in pd.read_parquet(part, columns=["image"]).image}
    scans = lance.dataset("runs/htr/curate/scans.lance"); model = sdm.models.KumoTabular(device="cuda"); start = time.time(); scored = 0
    for batch in scans.to_batches(columns=["image"] + NUM + CAT, filter="pool != 'gt' AND lines > 0", batch_size=BATCH):
        rows = tidy(batch.to_pandas()); rows = rows[~rows.image.isin(done)]
        if not len(rows): continue
        result = {"image": rows.image.to_numpy()}
        for target, kind, name in (("bad", "categorical", "quality_bad"), ("cer_matched", "numerical", "quality_cer")):
            df = pd.concat([context[NUM + CAT + [target]], rows[NUM + CAT].assign(**{target: context[target].iloc[0]})], ignore_index=True)
            t = sdm.TableTensor.from_pandas(df=df, stypes=sdm.infer_stypes(df, overrides={target: kind, **{c: "categorical" for c in CAT}}), device="cuda")
            with torch.amp.autocast("cuda", torch.float16):
                model.fit(x=t[:n].drop_columns(target), y=t[:n, target], num_estimators=8); got = model.predict(t[n:].drop_columns(target))
            result[name] = (got[:, "1"].numerical if kind == "categorical" else got.numerical.mean(dim=-1, keepdim=True)).float().cpu().numpy().reshape(len(rows), -1)[:, 0]
            model.clear()
        pd.DataFrame(result).to_parquet(out / f"part-{time.time_ns()}.parquet"); scored += len(rows)
        print(f"{scored} scans, {scored / (time.time() - start):.0f} a second", flush=True)
        if limit and scored >= limit: break
