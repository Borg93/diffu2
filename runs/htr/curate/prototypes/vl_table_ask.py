"""Does a sharper question make the vision-language model tell a table from running text (CURATION E19)?

The scans are the 36 looked at on 15 Nov from the model's "table or ledger" and "table across the fold" answers (the
first 32,864 described scans, 18 drawn from each with random_state 3), with what the eye said of each. Each is asked
one question, with and without thinking, and the same question as the first of a two-question form.

    uv run python runs/htr/curate/prototypes/vl_table_ask.py
"""
import base64, io, json, re, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
SEEN = 32864
LAST: list[str] = []
TABLES = {"table": {0, 2, 3, 4, 5, 8, 14, 17}, "table_across": {1, 4, 6, 7, 8, 9, 11, 12, 15, 17}}  # by eye; the rest prose
UNSURE = {("table_across", 2)}
ASK = ("This is a scan from a Swedish archive. Is the handwriting on it laid out as a TABLE or LEDGER, or as RUNNING TEXT?\n"
       "- table or ledger: ruled columns or rows, or short entries with numbers or amounts lined up in columns down the page.\n"
       "- running text: sentences in lines across the page, as in a letter, a court record or minutes, even if dense, "
       "with a margin, headings, dates or a few numbers.\n"
       'Return only JSON: {"layout": "table" | "running_text", "why": "at most 12 words"}')


def ask(image: str, think: bool) -> str:
    with Image.open(image) as scan:
        scan.draft("RGB", (3072, 3072)); scan = scan.convert("RGB"); scan.thumbnail((1536, 1536)); held = io.BytesIO(); scan.save(held, "JPEG", quality=85)
    body = {"model": "nemotron-3.5-super-vl", "temperature": 0.6, "top_p": 0.95, "max_tokens": 4096 if think else 200, "chat_template_kwargs": {"enable_thinking": think},
            "messages": [{"role": "user", "content": [{"type": "text", "text": ASK}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(held.getvalue()).decode()}}]}]}
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8011/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as answer:
        text = json.load(answer)["choices"][0]["message"]["content"] or ""
    LAST.append(text)
    found = re.search(r'"layout"\s*:\s*"(table|running_text)"', text)
    return found.group(1) if found else "none"


d = pd.read_json("runs/htr/curate/vl/sample.jsonl", lines=True); d = d[d.kind.notna()].iloc[:SEEN]
groups = {"table": d[d.kind == "handwritten_table_or_ledger"], "table_across": d[d.table_across_fold == 1]}
rows = [(name, j, r.image, j in TABLES[name]) for name, g in groups.items() for j, r in enumerate(g.sample(18, random_state=3).itertuples()) if (name, j) not in UNSURE]
ASK_ALONE = ASK
ASK_FORM = ("This is a scan from a Swedish archive. Answer two questions about the handwriting on it.\n"
            "1. writing_layout: is it laid out as a table or ledger, or as running text?\n"
            "   - table: ruled columns or rows, or short entries with numbers or amounts lined up in columns down the page.\n"
            "   - running_text: sentences in lines across the page, as in a letter, a court record or minutes, even if dense, with a margin, headings, dates or a few numbers.\n"
            "   - none: no handwriting (a sign, a cover, a blank).\n"
            "2. turned_writing: is any handwriting turned on its side or upside down against the rest?\n"
            "   - none: all writing runs the same way, upright.\n"
            "   - some: a note, an address or a margin entry is turned, the rest is upright.\n"
            "   - most: most of the writing is on its side or upside down as the scan lies.\n"
            'Return only JSON: {"layout": "table" | "running_text" | "none", "turned_writing": "none" | "some" | "most"}')
for think, ASK in ((False, ASK_ALONE), (True, ASK_ALONE), (False, ASK_FORM)):
    print("the question alone" if ASK is ASK_ALONE else "the two-question form", end=", ")
    with ThreadPoolExecutor(12) as pool: said = list(pool.map(lambda row: ask(row[2], think), rows))
    hit = sum((s == "table") == truth for s, (_, _, _, truth) in zip(said, rows))
    print(f"thinking {think}: right {hit} of {len(rows)}; tables called tables {sum(s == 'table' and t for s, (_, _, _, t) in zip(said, rows))} of {sum(t for *_, t in rows)}; prose called running text {sum(s == 'running_text' and not t for s, (_, _, _, t) in zip(said, rows))} of {sum(not t for *_, t in rows)}; no answer {said.count('none')}", flush=True)
    print("  wrong:", [(n, j, s) for s, (n, j, _, t) in zip(said, rows) if (s == "table") != t])
