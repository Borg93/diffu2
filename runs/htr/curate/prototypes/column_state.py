"""Where each column asked for stands, read from the tables themselves (CURATION, "Columns asked for").

WANTED is the list agreed with Gabriel on 17 Nov: a line a column of the one dataset every training set is filtered
from. For each, how many pool scans hold a value in scans.lance; a column the table does not have is "missing".
Add a line here the day a column is asked for; the ledger's table is this script's output.

    uv run python runs/htr/curate/prototypes/column_state.py
"""
from pathlib import Path

import lance

HOME = Path("runs/htr/curate")
WANTED = {  # column of scans.lance: (what it says, what makes it)
    "page_confidence": ("the old pipeline's confidence", "diffu-curate features"),
    "polygon_cover": ("statistics of the old labels (lines, digits, cover ...)", "diffu-curate features"),
    "gutter_score": ("the small gutter network's opening score", "diffu-curate gutter"),
    "saturation": ("greyscale microfilm (near 0) or colour scan", "embed_pool.py (E40), with the vectors"),
    "vl_caption": ("the VL model's caption", "vl_describe.py (a sample: too slow for every scan)"),
    "vl_kind": ("the VL model's kind of scan", "vl_describe.py, then a head for every scan"),
    "vl_writing_layout": ("table or running text, by the VL model", "vl_ask2.py"),
    "is_table": ("table or running text for every scan, by a head", "pool_columns.py (E40), with table_p"),
    "kind": ("the kind of scan for every scan, by a head", "pool_columns.py (E40), with kind_p"),
    "layout": ("opening, single page ... for every scan, by a head", "pool_columns.py (E40), with layout_p"),
    "cut_class": ("the hand marks' class for every scan, by a head on the patch vectors", "pool_columns.py (E40), with cut_p"),
    "colour": ("a colour scan or greyscale microfilm", "pool_columns.py (E40), from saturation"),
    "vl_turned": ("turned writing, by the VL model", "vl_ask2.py"),
    "orientation": ("which way up the writing is, by the small network", "to do: orientation.py over the pools"),
    "mark": ("the class marked by hand (cut, no_cut, single, not_a_page, odd)", "marks_to_lance.py"),
    "head_cut": ("the hand marks' class by the head on Qwen's vectors (described scans only)", "cut_head.py"),
    "fold_x": ("where the fold is", "place_folds.py; for every scan after E40"),
    "fold_found": ("whether a fold line was found", "place_folds.py"),
    "cluster": ("the cluster on the map of 14,985 scans (50 clusters on Qwen's vectors)", "scalar_hirank.py; for every scan by a head once the clusters are named"),
    "cluster_name": ("the cluster's name", "to do: Toponymy"),
    "covered": ("writing layer: share of the writing under a label", "to do: writing_eval.py's layer over the pools"),
    "jev_page": ("Jev's score of the page's lines", "to do: first on the calibration scans"),
    "read_disagreement": ("our reader against the old label", "pages.lance holds it for 64,950 pages; to do on the trusted pages"),
    "dinov3": ("DINOv3's vector of the scan (2,048 numbers, 16-bit)", "embed_pool.py, joined by vectors_to_lance.py (E40)"),
    "tipsv2": ("TIPSv2's vector of the scan (2,048 numbers, 16-bit)", "embed_pool.py, joined by vectors_to_lance.py (E40)"),
    "qwen8b": ("Qwen3-VL-Embedding 8B's vector", "scan_embeddings.lance holds 130,733"),
    "pca": ("the first 64 principal components of the two models' vectors (82 % of the variance), 16-bit", "pool_columns.py (E40), heads/pca.npz"),
}

if __name__ == "__main__":
    scans = lance.dataset(str(HOME / "scans.lance")); held = set(scans.schema.names); pool = scans.count_rows("pool != 'gt'")
    print(f"| Column | What it says | Pool scans with a value (of {pool:,}) | Made by |\n|---|---|---|---|")
    for column, (what, by) in WANTED.items():
        state = f"{scans.count_rows(f'pool != \'gt\' AND {column} IS NOT NULL'):,}" if column in held else "**missing**"
        print(f"| `{column}` | {what} | {state} | {by} |")
