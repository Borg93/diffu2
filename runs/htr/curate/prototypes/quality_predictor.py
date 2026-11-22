"""A quality predictor on the scan features (CURATION E68): which model tells a bad old label from a good one.

    CUDA_VISIBLE_DEVICES=0 /mnt/scratch/gabriel/envs/sdm/bin/python runs/htr/curate/prototypes/quality_predictor.py

The table (built from the calibration scans: pool scans the ground truth also holds, the old label set against the
truth) has 3,651 scans of 83 volumes. A scan is "bad" where the old label's matched lines err on over a tenth of their
characters, or over a tenth of the true lines are missed, or over a tenth of the old lines lie astray. Five folds of
whole volumes. Compared: each feature alone, LightGBM, and NVIDIA's KumoTabular (in-context: the training folds are
shown, nothing is trained), each with and without the 64 PCA numbers of the image vector. Printed: AUC and average
precision for "bad", Spearman for the matched lines' CER.
"""
import warnings

warnings.filterwarnings("ignore")
import lightgbm as lgb, numpy as np, pandas as pd, sdm, torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

NUM = ["width", "height", "page_confidence", "regions", "lines", "empty_lines", "chars", "words", "digit_share", "line_height_median", "line_height_spread", "line_width_median", "chars_per_100px_median", "odd_density_lines", "polygon_cover", "repeated_lines", "longest_word_run", "compression", "gutter_score", "cut_p", "table_p", "kind_p", "layout_p", "saturation", "language_p", "written", "labelled", "covered", "filled"]
CAT = ["pool", "cut_class", "kind", "layout", "language", "pool_cluster"]
PCA = [f"pca{k:02d}" for k in range(64)]
m = pd.read_parquet("/mnt/scratch/gabriel/htr/quality/calibration.parquet"); m["colour"] = m.colour.astype("float"); NUM = NUM + ["colour"]
import glob, json
from pathlib import Path
jev = pd.DataFrame([json.loads(line) for part in glob.glob("/mnt/scratch/gabriel/htr/quality/jev/part-*.jsonl") for line in open(part)])
jev["scan"] = jev.image.map(lambda i: Path(i).stem); JEV = ["jev_mean", "jev_low", "jev_worst"]
m = m.merge(jev.rename(columns={"mean": "jev_mean", "low": "jev_low", "worst": "jev_worst"})[["scan"] + JEV], on="scan", how="left")
print("scans with Jev:", int(m.jev_mean.notna().sum()), "of", len(m), flush=True)
for c in CAT: m[c] = m[c].astype("string").fillna("none").astype("category")
folds = list(GroupKFold(5).split(m, groups=m.volume))


def kumo(train, test, columns, target, kind):
    df = pd.concat([train, test])[columns + [target]].reset_index(drop=True)
    for c in CAT:
        if c in df: df[c] = df[c].astype(str)
    t = sdm.TableTensor.from_pandas(df=df, stypes=sdm.infer_stypes(df, overrides={target: kind, **{c: "categorical" for c in CAT if c in df}}), device="cuda")
    model = sdm.models.KumoTabular(device="cuda"); n = len(train)
    with torch.amp.autocast("cuda", torch.float16):
        model.fit(x=t[:n].drop_columns(target), y=t[:n, target], num_estimators=8); out = model.predict(t[n:].drop_columns(target))
    # a column is asked for by its name: the names' order is not the order of out.numerical's columns
    values = (out[:, "1"].numerical if kind == "categorical" else out.numerical.mean(dim=-1, keepdim=True)).float().cpu().numpy().reshape(len(test), -1)[:, 0]
    model.clear()
    return values


def boosted(train, test, columns, target, kind):
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    model = (lgb.LGBMClassifier if kind == "categorical" else lgb.LGBMRegressor)(**params).fit(train[columns], train[target])
    return model.predict_proba(test[columns])[:, 1] if kind == "categorical" else model.predict(test[columns])


rows = []
for name, fn, columns in (("KumoTabular, features", kumo, NUM + CAT), ("KumoTabular, features + Jev", kumo, NUM + CAT + JEV), ("KumoTabular, Jev only", kumo, JEV), ("LightGBM, features + Jev", boosted, NUM + CAT + JEV), ("KumoTabular, features + Jev + image", kumo, NUM + CAT + JEV + PCA)):
    bad, cer = np.zeros(len(m)), np.zeros(len(m))
    for tr, te in folds:
        bad[te] = fn(m.iloc[tr], m.iloc[te], columns, "bad", "categorical"); cer[te] = fn(m.iloc[tr], m.iloc[te], columns, "cer_matched", "numerical")
    per = [roc_auc_score(m.bad.iloc[te], bad[te]) for _, te in folds]
    rows.append((name, roc_auc_score(m.bad, bad), min(per), average_precision_score(m.bad, bad), spearmanr(m.cer_matched, cer).statistic)); m[f"p_{name}"] = bad
    print(f"{name:32} bad: AUC {rows[-1][1]:.3f} (worst fold {rows[-1][2]:.3f}), average precision {rows[-1][3]:.3f} | CER: Spearman {rows[-1][4]:.3f}", flush=True)
single = sorted(((max(a := roc_auc_score(m.bad, m[c].fillna(m[c].median())), 1 - a), c) for c in NUM + JEV), reverse=True)[:6]
print("one feature alone (AUC):", ", ".join(f"{c} {a:.3f}" for a, c in single), f"| bad pages: {m.bad.mean():.1%} of {len(m)}")
best = max(rows, key=lambda r: r[1])[0]; p = m[f"p_{best}"]; order = np.argsort(-p.to_numpy()); found = np.cumsum(m.bad.to_numpy()[order]) / m.bad.sum()
print(f"{best}: dropping the worst-scored 10 % of scans removes {found[len(m) // 10]:.0%} of the bad ones, 20 % removes {found[len(m) // 5]:.0%}, 30 % removes {found[3 * len(m) // 10]:.0%}")
m[["scan", "pool", "volume", "bad", "cer_matched", "missed_share", "astray_share"] + [c for c in m if c.startswith("p_")]].to_parquet("/mnt/scratch/gabriel/htr/quality/predictions_jev.parquet")
