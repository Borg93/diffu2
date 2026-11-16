"""Does the vision-language model decide better whether a scan is to be cut when shown marked examples (CURATION E32)?

The 200 scans marked by hand (fold_marks.lance): 167 with a fold, 27 not to be cut, 6 single pages. Each is asked
twice of the served model, itself left out: with the three answers defined only, and with the same definitions and
SHOTS marked scans of each answer shown first with their answers (drawn from the other 199, seeded by the scan).

    uv run python runs/htr/curate/prototypes/vl_cut_or_not.py [PORT]
"""
import base64, io, json, random, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HOME = Path("runs/htr/curate"); SIDE, SHOTS, WORKERS = 1024, 2, 6
URL = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else '8011'}/v1/chat/completions"
ASK = ("A scan from a Swedish archive is to be turned into single pages for a handwriting reader. Decide what to do with it.\n"
       "- cut: an open book or folded sheet showing two pages side by side, each page with its own text; cutting down the fold gives two pages that each read on their own. Columns of a ledger that stay within one page are still cut.\n"
       "- no_cut: two pages or a wide sheet where one table, one ledger row or one text runs across the fold, so that a line starts on the left page and continues on the right; cutting would split lines in two.\n"
       "- single: one page only (a single sheet, or one page of a book with at most a sliver of the facing page).\n"
       'Return only JSON: {"answer": "cut" | "no_cut" | "single"}')
SAID = {"fold": "cut", "no_cut": "no_cut", "single": "single"}


def picture(image: str) -> dict:
    with Image.open(image) as scan:
        scan.draft("RGB", (SIDE * 2, SIDE * 2)); scan = scan.convert("RGB"); scan.thumbnail((SIDE, SIDE)); held = io.BytesIO(); scan.save(held, "JPEG", quality=85)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(held.getvalue()).decode()}}


def ask(job: tuple[str, str, list[tuple[str, str]]]) -> dict:
    image, arm, shown = job
    content = [{"type": "text", "text": ASK}]
    for number, (example, answer) in enumerate(shown, 1):
        content += [{"type": "text", "text": f"Example {number}:"}, picture(example), {"type": "text", "text": json.dumps({"answer": answer})}]
    content += [{"type": "text", "text": "The scan to decide:" if shown else ""}, picture(image)]
    body = {"model": "nemotron-3.5-super-vl", "temperature": 0.0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}, "messages": [{"role": "user", "content": content}]}
    try:
        with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as reply:
            text = json.load(reply)["choices"][0]["message"]["content"] or ""
        return {"image": image, "arm": arm, "answer": json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())["answer"]}
    except (OSError, ValueError, KeyError) as error:
        return {"image": image, "arm": arm, "error": f"{type(error).__name__}: {error}"[:200]}


if __name__ == "__main__":
    marks = lance.dataset(str(HOME / "fold_marks.lance")).to_table().to_pandas().sort_values("marked_at").drop_duplicates("image", keep="last")
    truth = {row.image: SAID[row.mark] for row in marks.itertuples() if row.mark in SAID}
    jobs = []
    for image in truth:
        draw = random.Random(image); shown = []
        for answer in ("cut", "no_cut", "single"):
            shown += [(other, answer) for other in draw.sample(sorted(o for o, a in truth.items() if a == answer and o != image), SHOTS)]
        draw.shuffle(shown); jobs += [(image, "definitions", []), (image, "examples", shown)]
    with ThreadPoolExecutor(WORKERS) as pool, (HOME / "vl" / "cut_or_not.jsonl").open("w", encoding="utf-8") as out:
        for count, row in enumerate(pool.map(ask, jobs), 1):
            out.write(json.dumps({**row, "truth": truth[row["image"]]}) + "\n"); out.flush()
            if count % 40 == 0: print(count, "of", len(jobs), flush=True)
    print("done", flush=True)
