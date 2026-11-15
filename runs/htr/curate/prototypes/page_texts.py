"""The labelled text of each sampled page (the lines of its label file whose middle lies in the page's rectangle), for
embedding a page with its label."""
import json, sys
from multiprocessing import Pool
import pandas as pd
from diffu_page.page.layout.xml_replay import load_xml_layout
def text(row):
    try:
        layout = load_xml_layout(__import__("pathlib").Path(row["labels"]))
    except Exception:
        return ""
    return "\n".join(l.text for l in layout.lines if row["x0"] <= l.middle_x < row["x1"])
if __name__ == "__main__":
    import lance
    s = pd.read_json("runs/htr/curate/embed/sample.json")
    lab = lance.dataset("runs/htr/curate/pages.lance").to_table(columns=["page", "labels"], filter="error IS NULL").to_pandas()
    s = s.merge(lab, on="page", how="left")
    with Pool(24) as p: texts = p.map(text, s.to_dict("records"), chunksize=64)
    json.dump(texts, open("runs/htr/curate/embed/texts.json", "w"), ensure_ascii=False)
    print(len(texts), "empty", sum(1 for t in texts if not t), "median chars", sorted(len(t) for t in texts)[len(texts) // 2])
